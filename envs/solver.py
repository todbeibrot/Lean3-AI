from envs.proof_gen import ProofGenEnv
import numpy as np
from os.path import join
from utils.lean_gym import LeanGym
import gensim
import gym
from utils.scripts import get_theorems


class Solver(ProofGenEnv, gym.Env):

    def __init__(
        self,
        file_path: str,
        dir_path: str,
        image_len: int = 15,
        image_lines: int = 15,
        max_steps: int = 30,
        verbose: int = 0,
        max_variables: int = 30,
        max_intro_variables: int = 5,
        max_goals: int = 100,
        max_number: int = 100,
    ):
        gym.Env.__init__(self)
        self.file_path = file_path
        self.dir_path = dir_path
        self.verbose = verbose
        word2vec, word2vec_dim, default_value, vocabulary, max_word_lenght, low, high = self.load_word2vec()
        self.lean = LeanGym(dir_path, '', verbose, solve_sorry=True)
        self.get_declarations()
        ProofGenEnv.__init__(
            self,
            dir_path,
            self.lean,
            None,
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
            False,
            max_variables=max_variables,
            max_intro_variables=max_intro_variables,
            max_number=max_number,
            max_goals=max_goals,
            agent_id=0,
            evaluation_mode=True
        )
        self.infos = {
            'solved_theorems': 0,
            'attempts': 0
        }
        self.finished = False
        self.declaration_counter = 0
        for i in range(len(self.tactics)):
            if self.tactics[i].startswith('my_'):
                self.tactics[i] = self.tactics[i].removesuffix('my_')
        
        self._action_space_in_preferred_format = self.action_space
        self._obs_space_in_preferred_format = self.observation_space

    def reset(self):
        ProofGenEnv.reset(self)
        self.update_lean_infos(self.lean.init_search(self.get_decl()))
        self.update_goals()
        self.update_obs()
        return self.obs
    
    def step(self, action):
        obs, reward, done, finished_proof, _, _, _, _, _, _ = ProofGenEnv.step(self, action)
        if done:
            self.update_info(finished_proof)
        if self.finished:
            return obs, reward, done, {'finished': True}
        return obs, reward, done, {}

    def render(self, mode: str = 'human'):
        print(self.infos)

    def close(self):
        self.lean.close()
        super().close()

    def load_word2vec(self):
        # Load Word2Vec model
        word2vec = gensim.models.Word2Vec.load(join(self.dir_path, 'data', 'word2vec')).wv
        vocabulary = word2vec.index_to_key
        word2vec_dim = word2vec[0].shape[0]
        default_value = np.zeros((word2vec_dim,), dtype=np.float32)

        # Get lowest and highest value for the limits of action space and observation space
        low = np.inf
        high = -np.inf
        max_word_lenght = 0
        for word in vocabulary:
            if len(word) > max_word_lenght:
                max_word_lenght = len(word)
            for value in word2vec[word]:
                if value < low:
                    low = value
                if value > high:
                    high = value
        return word2vec, word2vec_dim, default_value, vocabulary, max_word_lenght, low, high
    
    def get_decl(self):
        self.declaration = self.declarations[self.declaration_counter]
        self.declaration_counter = (self.declaration_counter + 1) % len(self.declarations)
        return self.declaration
    
    def get_declarations(self):
        self.declarations = get_theorems(self.file_path)

    def update_info(self, finished_proof: bool):
        self.infos['solved_theorems'] += finished_proof
        self.infos['attempts'] += 1
        if self.infos['attempts'] == 244:
            self.finished = True
        if self.verbose > 0:
            print(self.infos)
