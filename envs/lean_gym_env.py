import numpy as np
from random import randint
from re import sub
from typing import Dict, List, Optional, Tuple
from gymnasium.spaces import Discrete, MultiDiscrete, Box
import gymnasium.spaces
from utils.lean_gym import LeanGym


# For communication with lean-gym and handling of the observations and stuff
class LeanGymEnv():


    def __init__(self,
        dir_path: str,
        lean_gym: LeanGym,
        corpus,
        word2vec,
        word2vec_dim,
        default_value: np.ndarray,
        image_len: int = 15,
        image_lines: int = 30,
        vocabulary: List = [],
        low: float = 15.,
        high: float = 15.,
        max_steps: int = 30,
        verbose: int = 0,
        generate_lemmas: bool = False,
        generate_proofs: bool = False,
        allow_cheats: bool = False,
        max_variables: int = 30,
        max_intro_variables: int = 10,
        max_number: int = 100,
        max_goals: int = 100,
        evaluation_mode: bool = False,
        n_actions: int = 1
    ):
        self.dir_path = dir_path
        self.verbose = verbose
        self.lean = lean_gym
        self.word2vec = word2vec
        self.corpus = corpus
        self.generate_proofs = generate_proofs
        self.generate_lemmas = generate_lemmas
        self.allow_cheats = allow_cheats
        self.word2vec_dim = word2vec_dim
        self.max_steps = max_steps
        self.default_value = default_value
        self.vocabulary = vocabulary
        self.image_len = image_len
        self.image_lines = image_lines
        self.evaluation_mode = evaluation_mode
        self.max_number = max_number
        self.agent_id = 0

        self.write_corpus = bool(corpus)

        # Observation
        # Max number of variables
        self.max_variables = max_variables
        self.max_intro_variables = max_intro_variables
        # If there are more goals the model sees it as max_goals goals
        self.max_goals = max_goals

        self.proof_steps = ''
        self.tactic_state = ''
        self.error = None
        self.tactic_state_id = 0
        self.search_id = 0
        self.done = False
        self.n_used_intro_variables = 0
        self.n_goals = 0
        self.first_action = True

        # Observation space
        # The image will be an image of the tactic state. 
        image = Box(low=low, high=high, shape=(self.image_lines, self.image_len, self.word2vec_dim), dtype=np.float32)
        n_goals = Box(low=0, high=self.max_goals, shape=(1,), dtype=np.float32)
        steps = Box(low=0, high=self.max_steps, shape=(1,), dtype=np.float32)
        # TODO: make Discrete. Doesn't work with rllib
        first_action = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        action_mask = Box(0, 1, shape=(n_actions, ), dtype=np.float32)
        if generate_lemmas:
            features = gymnasium.spaces.Dict({
                'n_goals': n_goals,
                'steps': steps
            })
        else:
            features = gymnasium.spaces.Dict({
                'n_goals': n_goals,
                'steps': steps,
                'first_action': first_action
            })
        self.observation_space = gymnasium.spaces.Dict({
            'image': image,
            'features': features,
            'action_mask': action_mask
        })
        self.image = np.zeros((self.image_lines, self.image_len, self.word2vec_dim), dtype=np.float32)
        self.obs = self.observation_space.sample()
        # The tactic state as it will be given as image in the observation. 
        self.processed_tactic_state = [['']]
        # If an action leads to a tactic with an error, we don't want to repeat it
        self.masked_actions = []
        self.action = 0
        self.masked_actions_obs = np.zeros((n_actions, ), dtype=np.float32)

    def reset(self):
        self.proof_steps = ''
        self.tactic_state = ''
        self.error = None
        self.tactic_state_id = 0
        self.search_id = 0
        self.done = False
        self.steps = 0
        self.masked_actions = []

        self.n_goals = 1
        self.n_used_intro_variables = self.max_variables
        if self.generate_lemmas:
            self.generated_new_lemma = False
            self.start_id = 0
        if self.generate_proofs:
            self.finished_proof = False
    
    def step(self, action):
        self.steps += 1
        if self.steps == self.max_steps:
            self.done = True
        self.action = action

    def close(self):
        pass

    def run_tactic(self, tactic: str):
        self.run_tac(self.search_id, self.tactic_state_id, tactic)
        if self.error:
            self.debugging_errors()
        self.update_masked_actions()

    def plan_run_tactic(self, tactic: str):
        self.lean.plan_run_tactic(self.search_id, self.tactic_state_id, tactic, self.agent_id)
    
    def run_tac(self, search_id: int, tactic_state_id: int, tactic: str):
        self.update_lean_infos(self.lean.run_tac(search_id, tactic_state_id, tactic))

    def update_lean_infos(self, lean_infos: Dict):
        self.error = lean_infos['error']
        if lean_infos['search_id']:
            self.search_id = int(lean_infos['search_id'])
        if lean_infos['tactic_state_id']:
            self.tactic_state_id = int(lean_infos['tactic_state_id'])
        if lean_infos['tactic_state']:
            self.tactic_state = str(lean_infos['tactic_state'])
        if lean_infos['proof_steps']:
            self.proof_steps = str(lean_infos['proof_steps'])

        if self.error and self.error.startswith('unknown id'):
            self.done = True

    def check_done(self):
        raise NotImplementedError

    def append_word2vec_file(self, tactic_state):
        for line in tactic_state:
            self.corpus.write(' '.join(line) + '\n')

    def translate_to_numbers(self, word: str):
        if word in self.vocabulary:
            return self.word2vec[word]
        return self.default_value

    # Splits the tactic state into Tokens
    def preprocess_tactic_state(self, tactic_state: str) -> List[List[str]]:
        return [self.preprocess_line(line) for line in tactic_state.split('\n') if 'goals' not in line]

    def preprocess_line(self, line: str):
        # Adding spaces around commas and parentheses cause we want to handle them as words
        line_list = (sub(' +', ' ', line.replace(',\n', '\n').replace(',', ' , ').replace('(', ' ( ').replace(')', ' ) ')\
           .replace('{', ' { ').replace('}', ' } ').replace('[', ' [ ').replace(']', ' ] ').replace('+', ' + '))).strip().split()
        # Insert an '&' between numbers, so they don't merge
        inserted = 0
        for i in range(len(line_list) - 1):
            if line_list[i + inserted].isnumeric() and line_list[i + inserted + 1].isnumeric():
                line_list.insert(i + inserted + 1, '&')
                inserted += 1
        # Break the numbers into digits
        new_list = []
        for i in line_list:
            if i.isnumeric():
                new_list += list(i)
            else:
                new_list.append(i)
        return new_list
    
    def find_used_intro_variables(self):
        self.n_used_intro_variables = self.max_variables
        while f'h{self.n_used_intro_variables}' in self.tactic_state:
            self.n_used_intro_variables += 1

    def update_obs(self):
        if self.tactic_state == 'no goals':
            return
        if not self.error:
            tactic_state = self.preprocess_tactic_state(self.tactic_state)
            if self.write_corpus:
                self.append_word2vec_file(tactic_state)
            self.processed_tactic_state = [line[:self.image_len] for line in tactic_state[:self.image_lines]]

            # Observation
            self.image[:][:] = self.default_value
            for _line, line in enumerate(self.processed_tactic_state):
                for token in range(len(line)):
                    self.image[_line][token] = self.translate_to_numbers(self.processed_tactic_state[_line][token])
        features = {
            'n_goals': np.array([self.n_goals], dtype=np.float32),
            'steps': np.array([self.steps], dtype=np.float32)
        }
        if not self.generate_lemmas:
            features['first_action'] = np.array([self.first_action], dtype=np.float32)
        self.masked_actions_obs[:] = 0
        for action in self.masked_actions:
            self.masked_actions_obs[action] = 1
        self.obs = {
            'image': self.image,
            'features': features,
            'action_mask': self.masked_actions_obs
        }

    def update_goals(self):
        goal_line = self.tactic_state.split('\n')[0]
        # Get number of goals
        if goal_line == 'no goals':
            self.n_goals = 0
        elif 'goals' in goal_line:
            try:
                self.n_goals = max(0, min(int(sub('\\D', '', goal_line)), self.max_goals))
            except:
                self.done = True
                self.error_message(self, 'failed to read number of goals', 0)
        else:
            self.n_goals = 1

    def update_masked_actions(self):
        if self.error:
            self.masked_actions.append(self.action)
        else:
            self.masked_actions = []

    # Returns the word at the position in the tactic state
    def get_word_at_position(self, position) -> str:
        position_line = (position - (position % self.image_len)) % self.image_lines
        position_token = position % self.image_len
        if len(self.processed_tactic_state) > position_line and len(self.processed_tactic_state[position_line]) > position_token:
            return self.processed_tactic_state[position_line][position_token]
        return ''
    
    # Close the current goal by adding it to the variables und apply it
    def close_main_goal(self):
        assert self.allow_cheats
        self.run_tactic('close_goal')

    # Adds a new variable to the tactic state. It will focus a goal of type 'Sort *'. The proof for this goal will be the type of the new variable.
    def add_new_variable(self):
        assert self.allow_cheats
        goals = self.tactic_state.split('\n\n')
        # n will be used for 'swap n'
        n = 1
        # Find a goal to determine the type of the new variable
        for goal in goals:
            if 'ccc_goal' in goal[goal.find('|-'):]:
                break
            else:
                n += 1
        if n > 1:
            self.run_tactic(f'swap {n}')
        




    





    # Debugging

    def error_message(self, msg, verbose):
        if verbose < 1:
            return
        print(30 * '+')
        print(msg)
        if verbose < 3:
            return
        print(f'error: {self.error}')
        print(f'search_id: {self.search_id}')
        print(f'tactic_state_id: {self.tactic_state_id}')
        print('tactic_state:')
        print(self.tactic_state)
        

    def debugging_errors(self):
        # if 'unknown identifier' in self.error:
        #     self.error = self.error.removeprefix("gen_tac_and_capture_res_failed: pos=(some ⟨1, 2⟩) msg=unknown identifier 'h")
        #     self.error = self.error.removeprefix("gen_tac_and_capture_res_failed: pos=(some ⟨1, 2⟩) msg=unknown identifier 'b'")
        #     self.error = self.error.removesuffix("'")
        #     self.error = self.error.removesuffix("'")
        #     try:
        #         if self.error and not self.error.isnumeric() and self.error != 'new_lemma':
        #             self.error_message('unknown identifier', 8)
        #     except:
        #         print('unknown identifier and an error in my code: ' + self.error)
        pass
