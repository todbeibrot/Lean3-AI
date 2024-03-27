import gymnasium as gym
import numpy as np
from os.path import join
import time
from random import randint, random
import gensim
from envs.lemma_gen import LemmaGenEnv
from envs.proof_gen import ProofGenEnv
from utils.lean_gym import LeanGym
from gymnasium.spaces import Discrete
from ray.rllib.env import MultiAgentEnv
from typing import Tuple, Dict
import wandb


class SolverTrainEnv(MultiAgentEnv):

    # This env contains 3 envs:
    # - lemmagen: an env to generate lemmas
    # - proof_gen: an env to generate a lemma with a proof for a given lemma
    # - solver: an env to proof a given lemma

    def __init__(
        self,
        dir_path: str,
        use_wandb_logging: bool = True,
        #wandb_run = None,
        lean_reset_steps: int = 200000,
        image_len: int = 15,
        image_lines: int = 30,
        lemmagen_max_steps: int = 30,
        solver_max_steps: int = 30,
        max_variables: int = 8,
        max_intro_variables: int = 10,
        max_goals: int = 100,
        max_number: int = 100,
        verbose: int = 0,
        n_solvers: int = 1,
        generate_lemma_reward: float = 0.,
        n_envs: int = 2,
        id: int = 0,
        run_name: str = 'run_name',
        project_name: str = 'solver',
        continue_training: bool = False,
        solve_sorrys: int = 0,
        load_thm_path: str = ''
    ):
        self.dir_path = dir_path
        self.verbose = verbose
        self.use_wandb = use_wandb_logging
        #self.wandb_run = wandb_run
        self.n_solvers = n_solvers
        self.lean_reset_steps = lean_reset_steps
        self.generate_lemma_reward = generate_lemma_reward
        self.masked_action_penalty_lemmagen = 1 / lemmagen_max_steps
        self.masked_action_penalty_solver = 1 / solver_max_steps
        self.n_envs = n_envs
        self.id = id
        self.solve_sorry = bool(solve_sorrys)
        self.n_sorrys = 0
        self.start_n_sorrys = solve_sorrys
        self.sorrys_solved = 0
        if not load_thm_path:
            load_thm_path = join(self.dir_path, 'data', 'thms.txt')

        self.lean = LeanGym(dir_path, 'lemma_gen', verbose, solve_sorry=self.solve_sorry, n_solvers=n_solvers, thm_path=load_thm_path)
        self.lean.set_start_tactic(
            ','.join([
                f'let h{variable}_type : ccc_goal Sort*, rotate, have h{variable} : h{variable}_type := sorry, dsimp [h{variable}_type] at h{variable}, clear h{variable}_type'
                for variable in reversed(range(max_variables))
            ]) + ', fapply my_struc.mk'
        )

        # Load Word2Vec model
        word2vec, word2vec_dim, default_value, vocabulary, low, high = self.load_word2vec()

        # File to write down tactic states, so that we can train word2vec afterwards
        corpus_path = join(dir_path, 'data', 'corpi', f'corpus_{time.time()}_{randint(0, 2**15)}.txt')
        self.corpus = open(corpus_path, 'w', encoding='utf-8')
        
        self.lemmagen = LemmaGenEnv(
            dir_path,
            self.lean,
            self.corpus,
            word2vec,
            word2vec_dim,
            default_value,
            image_len,
            image_lines,
            vocabulary,
            low,
            high,
            lemmagen_max_steps,
            verbose,
            max_variables=max_variables,
            max_intro_variables=max_intro_variables,
            max_number=max_number,
            max_goals=max_goals
        )
        self.solvers = {}
        self.solver_agents = set([f'solver{i}' for i in range(self.n_solvers)])
        self._agent_id = self.agents = {'lemmagen'} | self.solver_agents
        self._action_space_in_preferred_format = gym.spaces.Dict({'lemmagen': self.lemmagen.action_space})
        self._obs_space_in_preferred_format = gym.spaces.Dict({'lemmagen': self.lemmagen.observation_space})
        for agent in range(self.n_solvers):
            name = f'solver{agent}'
            self.solvers[name] = ProofGenEnv(
                dir_path,
                self.lean,
                self.corpus,
                word2vec,
                word2vec_dim,
                default_value,
                image_len,
                image_lines,
                vocabulary,
                low,
                high,
                solver_max_steps,
                verbose,
                allow_cheats=False,
                max_variables=max_variables,
                max_intro_variables=max_intro_variables,
                max_number=max_number,
                max_goals=max_goals,
                agent_id=agent,
                evaluation_mode=self.solve_sorry
            )
            self._action_space_in_preferred_format[name] = self.solvers[name].action_space
            self._obs_space_in_preferred_format[name] = self.solvers[name].observation_space
        self.observation_space = self.lemmagen.observation_space
        self.action_space = self.lemmagen.action_space
        # Load lemma stands for reusing generated lemmas to train the solvers.
        self.load_lemma_proportion = 1.
        # Load theorem uses theorems of mathlib to train the solvers
        self.load_theorem_proportion = 0.1
        self.lean_gym_needs_reset = False
        self.n_steps = 0
        self.n_successful_agents = 0
        self.n_episodes = 0
        self.n_episodes_lemmagen = 0
        self.n_episodes_solver = 0
        self.agent_steps = 0
        super().__init__()
        if self.use_wandb:
            wandb.init(
                name=run_name + f'_{self.id}',
                project=project_name,
                sync_tensorboard=False,  # auto-upload sb3's tensorboard metrics
                monitor_gym=False,  # auto-upload the videos of agents playing the game
                save_code=False,  # optional
                # config=env_kwargs
                resume=False,
                group=run_name
            )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.n_episodes += 1
        if self.lean_gym_needs_reset:
            self.reset_lean_gym()

        self.load_theorem = self.solve_sorry or (random() < self.load_theorem_proportion)
        self.load_lemma = (random() < self.load_lemma_proportion and self.lean.can_load_lemmas and not self.load_theorem)
        self.load_something = self.load_theorem or self.load_lemma
        self.generate_lemma = not self.load_lemma and not self.load_theorem
        self.n_successful_agents = 0
        if self.load_something:
            self.n_episodes_solver += 1
            self.active_agents = self.solver_agents.copy()
            if self.load_theorem:
                search_id, tactic_state_id, tactic_state = self.lean.load_theorem()
            elif self.load_lemma:
                search_id, tactic_state_id, tactic_state = self.lean.load_lemma()
            observations = {}
            infos = {}
            for agent in self.active_agents:
                    obs = self.solvers[agent].reset(search_id, tactic_state_id, tactic_state)
                    observations[agent] = obs
                    infos[agent] = {}
            return observations, infos
        else:
            self.n_episodes_lemmagen += 1
            self.active_agents = {'lemmagen'}
            return {'lemmagen': self.lemmagen.reset()}, {}

    def step(self, action_dict):
        self.n_steps += 1
        self.agent_steps += len(action_dict)
        if self.n_steps % self.lean_reset_steps == 0:
            self.lean_gym_needs_reset = True
        # Return values
        observations  = {}
        rewards  = {}
        terminated = {}
        truncated = {}
        infos = {}
        return_values = (observations, rewards, terminated, truncated, infos)
        
        if self.generate_lemma:
            obs, done, is_success, search_id, tactic_state_id, tactic_state, used_masked_action = self.lemmagen.step(action_dict['lemmagen'])
            self.add_return_values(return_values, 'lemmagen', obs, reward=is_success * self.generate_lemma_reward - used_masked_action * self.masked_action_penalty_lemmagen, terminated=done, truncated=False, info={})
            if done:
                self.active_agents.remove('lemmagen')
            if is_success:
                self.generate_lemma = False
                self.n_episodes_solver += 1
                self.active_agents = self.solver_agents.copy()
                for agent in self.active_agents:
                    obs = self.solvers[agent].reset(search_id, tactic_state_id, tactic_state)
                    observations[agent] = obs
                    infos[agent] = {}
        else:
            terminated_agents = set()
            for agent in self.active_agents:
                used_masked_action = self.solvers[agent].choose_tactic(action_dict[agent])
            self.lean.run_planned_tactics()
            for agent in self.active_agents:
                obs, done, is_success = self.solvers[agent].process_results()
                self.add_return_values(return_values, agent, obs, reward=is_success - used_masked_action * self.masked_action_penalty_solver, terminated=done, truncated=False, info={})
                if done:
                    terminated_agents.add(agent)
                if is_success:
                    self.n_successful_agents += 1
            self.active_agents -= terminated_agents

        all_done = not bool(self.active_agents)
        terminated['__all__'] = all_done
        truncated['__all__'] = False
        if all_done and not self.generate_lemma:
            rewards['lemmagen'] = self.lemmagen_reward(self.n_successful_agents, self.n_solvers) * bool(self.n_successful_agents) + self.generate_lemma_reward
        if all_done and self.solve_sorry:
            self.sorrys_solved += bool(self.n_successful_agents)
            self.n_sorrys += 1
            # if self.n_sorrys % self.start_n_sorrys == 0:
            #     print(f'solved {self.sorrys_solved} of {self.n_sorrys} theorems')
            #     print(f'that is {self.sorrys_solved / self.n_sorrys * 100}%')
        if all_done:
            self.log_infos()
        if all_done:
            self.lean.clear_search()
        # 'load_lemma_proportion' and converges to the fail rate of the solvers
        if all_done and not self.generate_lemma:
            self.load_lemma_proportion = self.load_lemma_proportion - 0.01 * (self.load_lemma_proportion - (1 - (self.n_successful_agents / self.n_solvers)))
        return return_values

    def add_return_values(self, return_values: Dict, agent: str, observation, reward, terminated, truncated, info):
        return_values[0][agent] = observation
        return_values[1][agent] = reward
        return_values[2][agent] = terminated
        return_values[3][agent] = truncated
        return_values[4][agent] = info

    def render(self, mode: str = 'human'):
        pass

    def close(self):
        self.lemmagen.close()
        for agent in self.solvers.values():
            agent.close()
        self.corpus.close()
        self.lean.close()

    def load_word2vec(self):
        # Load Word2Vec model
        word2vec = gensim.models.Word2Vec.load(join(self.dir_path, 'data', 'word2vec')).wv
        vocabulary = word2vec.index_to_key
        word2vec_dim = word2vec[0].shape[0]
        default_value = np.zeros((word2vec_dim,), dtype=np.float32)

        # Get lowest and highest value for the limits of action space and observation space
        low = np.inf
        high = -np.inf
        for word in vocabulary:
            for value in word2vec[word]:
                if value < low:
                    low = value
                if value > high:
                    high = value
        return word2vec, word2vec_dim, default_value, vocabulary, low, high
    
    # reward of lemmagen dependent on how many solvers are able to solve the lemma. 
    # Does not include the constant reward for generating a lemma
    # It should be 0 if every or no agent solve the lemma.
    # It should be high in the middle
    # The derivative should be high around 0 and 1 and low in the middle
    def lemmagen_reward(self, n_successful_agents, n_solvers):
        success_rate = n_successful_agents / n_solvers
        return 5 * success_rate * (1 - success_rate)

    def reset_lean_gym(self):
        # Reset lean-gym
        self.lean_gym_needs_reset = False
        self.lean.reset_lean_gym()

    def log_infos(self):
        info = {
            'agent_steps': self.n_envs * self.agent_steps,
            'episodes/episodes': self.n_envs * self.n_episodes,
            'episodes/lemmagen': self.n_envs * self.n_episodes_lemmagen,
            'episodes/solver': self.n_envs * self.n_episodes_solver,
            'load_lemma/load_lemma': self.load_lemma_proportion,
            'load_lemma/mathlib': self.load_theorem_proportion
        }
        success_rate = self.n_successful_agents / self.n_solvers
        if self.load_lemma:
            info['success/solver_load_lemma'] = success_rate
        elif self.load_theorem:
            info['success/solver_mathlib'] = success_rate
        else:
            info['success/lemmagen'] = not self.generate_lemma
            info['episode_lenght/lemmagen'] = self.lemmagen.steps
        if self.generate_lemma:
            info['reward/lemmagen'] = 0
        else:
            info['success/solver'] = success_rate
            info['episode_lenght/solver'] = np.mean([agent.steps for agent in self.solvers.values()])
        if not self.generate_lemma and not self.load_something:
            info['success/new_lemmas'] = success_rate
            info['reward/lemmagen'] = self.generate_lemma_reward + (1 - success_rate) * bool(self.n_successful_agents)
        if self.solve_sorry:
            info['trys'] = self.n_sorrys
            info['solved'] = self.sorrys_solved
            info['solved_proportion'] = self.sorrys_solved / self.n_sorrys

        for key, value in info.items():
            if isinstance(value, bool):
                info[key] = int(value)
        if self.use_wandb:
            wandb.log(info)


        