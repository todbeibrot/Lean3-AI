import numpy as np
from os.path import join
from gymnasium.spaces import Discrete
from typing import List
from utils.lean_gym import LeanGym
from envs.lean_gym_env import LeanGymEnv
import time
from random import randint


class LemmaGenEnv(LeanGymEnv):

    def __init__(self,
        dir_path: str,
        lean_gym: LeanGym,
        corpus,
        word2vec,
        word2vec_dim: int,
        default_value: np.ndarray,
        image_len: int = 15,
        image_lines: int = 30,
        vocabulary: List = [],
        low: float = 15.,
        high: float = 15.,
        max_steps: int = 30,
        verbose: int = 0,  
        max_variables: int = 30,
        max_intro_variables: int = 10,
        max_number: int = 100,
        max_goals: int = 100,
    ):
        self.defs = []
        with open(join(dir_path, 'data', 'defs.txt'), 'r', encoding='utf-8') as defs_file:
            for definition in defs_file:
                self.defs.append(definition.removesuffix('\n'))
        self.n_defs = len(self.defs)

        # possible actions: apply def, apply variable, (intro, add_variable, apply_instance, close_goal), 10 digits + not_used, apply number
        self.n_actions = self.n_defs + max_variables + 4 + 11 + 1
        self.action_space = Discrete(self.n_actions)

        self.smallest_variable = self.n_defs
        self.biggest_variable = self.n_defs + max_variables - 1
        self.intro = self.biggest_variable + 1
        self.add_variable = self.biggest_variable + 2
        self.apply_instance = self.biggest_variable + 3
        self.close_goal = self.biggest_variable + 4
        self.smallest_number = self.biggest_variable + 5
        self.biggest_number = self.smallest_number + max_number

        self.new_lemma_reward = 0.05

        # Files to write new lemmas down. For debugging. Has no influence on training
        self.n_new_lemmas = 0
        self.new_lemmas = open(join(dir_path, 'data', 'lemmas_with_proofs', f'lemmas_with_proofs_{time.time()}_{randint(0, 2**15)}.txt'), 'w', encoding='utf-8')
        
        super().__init__(
            dir_path,
            lean_gym,
            corpus,
            word2vec,
            word2vec_dim,
            default_value,
            image_len,
            image_lines,
            vocabulary,
            low,
            high,
            max_steps,
            verbose,
            generate_lemmas=True,
            allow_cheats=True,
            max_variables=max_variables,
            max_intro_variables=max_intro_variables,
            max_number=max_number,
            max_goals=max_goals,
            n_actions=self.n_actions
        )
        

    def reset(self):
        super().reset()
        self.update_lean_infos(self.lean.init_search())
        return self.obs

    def step(self, action: int):
        if self.done:
            return self.obs, self.done, self.generated_new_lemma, self.search_id, self.tactic_state_id, self.tactic_state, action in self.masked_actions
        super().step(action)

        if self.is_number(action):
            self.run_tactic(f'apply {self.get_number(action)}')
        elif action < self.n_defs:
                self.run_tactic('fapply ' + self.defs[action])
        elif action == self.intro:
            self.run_tactic(f'intro h{self.n_used_intro_variables}')
            self.n_used_intro_variables += 1
        elif action == self.apply_instance:
            self.run_tactic('apply_instance')
        elif self.is_variable(action):
            self.run_tactic(f'apply h{action - self.smallest_variable}')
        elif action == self.add_variable:
            self.add_new_variable()
        elif action == self.close_goal:
            self.close_main_goal()
        else:
            self.error_message('lemmagen action too big', 0)
        
        self.update_goals()
        self.check_done()
        self.update_obs()
        return self.obs, self.done, self.generated_new_lemma, self.search_id, self.tactic_state_id, self.tactic_state, action in self.masked_actions
        
    def close(self):
        super().close()

    def check_done(self):
        if 'aaa' in self.tactic_state.split('\n\n')[0]:
            self.run_tactic('any_goals {exact true}, clear_trivial')
            self.update_goals()
            if self.n_goals == 1:
                self.generated_new_lemma = True
                self.lean.save_lemma()
                self.write_lemma_down()
            self.done = True
        
    def is_variable(self, action):
        return self.smallest_variable <= action <= self.biggest_variable

    def is_number(self, action):
        return self.smallest_number <= action <= self.biggest_number

    def get_number(self, action):
        return action - self.smallest_number
    
    # Write generated lemmas (with proofs) down. For debugging. It has no influence on training.
    def write_lemma_down(self):
        self.new_lemmas.write(self.tactic_state.split('\n\n')[0] + '\n\n')
