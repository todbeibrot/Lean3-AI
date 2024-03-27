import numpy as np
from os.path import join
from gymnasium.spaces import Discrete
import time
from random import randint
from utils.lean_gym import LeanGym
from envs.lean_gym_env import LeanGymEnv
from typing import List, Tuple

class ProofGenEnv(LeanGymEnv):

    def __init__(
        self,
        dir_path: str,
        lean_gym: LeanGym,
        corpus,
        word2vec,
        word2vec_dim: int,
        default_value = np.ndarray,
        image_len: int = 15,
        image_lines: int = 30,
        vocabulary: List = [],
        low: float = 15.,
        high: float = 15.,
        max_steps: int = 30,
        verbose: int = 0,
        allow_cheats: bool = False,
        max_variables: int = 30,
        max_intro_variables: int = 10,
        max_number: int = 100,
        max_goals: int = 100,
        evaluation_mode: bool = False,
        agent_id: int = 0,
    ):
        self.agent_id = agent_id

        # Load theorems
        # We will use 'theorem' for theorems and lemmas in mathlib and 'lemma' for the generated lemmas and the generated proofs
        self.theorems = []
        with open(join(dir_path, 'data', 'theorems_by_hand.txt'), 'r', encoding='utf-8') as theorems_file:
            for theorem in theorems_file:
                self.theorems.append(theorem.removesuffix('\n'))
        self.n_theorems = len(self.theorems)

        # Tactics
        tactics_vanilla = [
            'symmetry', 'simp', 'simp at *', 'norm_num', 'norm_num at *', 'push_neg', 'apply_assumption', 'apply_instance', 'abel', 'my_ring', 'group', 'dec_trivial', 'split', 'left', 'right',
            'constructor', 'refine {..}', 'linarith', 'nlinarith', 'cc', 'refl', 'trivial', 'my_assumption', 'dsimp', 'dsimp at *', 'my_library_search', 'ext', 'ext1', 'field_simp', 'norm_cast',
            'norm_fin', 'positivity', 'qify', 'zify', 'ring_exp'
        ]
        if evaluation_mode:
            for id, tactic in enumerate(tactics_vanilla):
                if tactic.startswith('my_'):
                    tactics_vanilla[id] = tactic.removeprefix('my_')
        tactics_with_variable = ['intro <variable>']
        tactics_with_theorem = ['rw <theorem>', 'rw <theorem> at *', 'rw ← <theorem>', 'rw ← <theorem> at *', 'apply <theorem>', 'fapply <theorem>', 'simp only [<theorem>]']
        # The following tactics take as input a word which we determine by giving the position in the observation.
        # This way it is possible to unfold things in the local context.
        # tactics_with_position = ['cases <token>', 'fin_cases <token>', 'induction <token>', 'unfold <token>', 'contrapose <token>']
        tactics_with_position = []
        tactics_with_number = ['apply <number>']
        self.tactics = tactics_vanilla + tactics_with_variable + tactics_with_theorem + tactics_with_position + tactics_with_number

        self.n_tactics = len(self.tactics)

        # To determine the action
        self.smallest_variable = self.n_theorems
        self.biggest_variable = self.n_theorems + max_variables - 1
        
        # Action space
        self.actions = tactics_vanilla + tactics_with_variable
        for tactic in tactics_with_theorem:
            for theorem in self.theorems:
                self.actions.append(tactic.replace('<theorem>', theorem))
            for variable in range(max_variables + max_intro_variables):
                self.actions.append(tactic.replace('<theorem>', f'h{variable}'))
        for tactic in tactics_with_position:
            for _ in range(image_len * image_lines):
                self.actions.append(tactic)
        for tactic in tactics_with_number:
            for number in range(max_number):
                self.actions.append(tactic.replace('<number>', str(number)))
        self.n_actions = len(self.actions)
        self.action_space = Discrete(self.n_actions)
        
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
            generate_proofs=True,
            allow_cheats=allow_cheats,
            max_variables=max_variables,
            max_intro_variables=max_intro_variables,
            max_number=max_number,
            max_goals=max_goals,
            evaluation_mode=evaluation_mode,
            n_actions=self.n_actions
        )

    def reset(self, search_id: int = 0, tactic_state_id: int = 0, tactic_state: str = ''):
        super().reset()
        self.search_id = search_id
        self.tactic_state_id = tactic_state_id
        self.tactic_state = tactic_state
        if self.tactic_state == 'no goals':
            self.done = self.finished_proof = True
        # In the first action the model gets to decide if it wants to negate the goal. 
        # In evaluation mode we expect the goal to be solvable
        self.first_action = (not self.evaluation_mode and 'aaa' in self.tactic_state)
        self.start_id = self.tactic_state_id
        self.find_used_intro_variables()
        self.update_goals()
        self.update_obs()
        return self.obs

    def choose_tactic(self, action: int):
        self.used_masked_action = (action in self.masked_actions)
        if self.used_masked_action:
            return True
        if self.done:
            return False
        super().step(action)
        tactic = self.actions[action]

        if self.first_action:
            self.first_action = False
            negated = (action % 2 == 0)
            if negated:
                self.plan_run_tactic('rw aaa_unfold_neg')
            else:
                self.plan_run_tactic('rw aaa_unfold')
        else:
            if '<variable>' in tactic:
                self.n_used_intro_variables += 1
                tactic = tactic.replace('<variable>', f'h{self.n_used_intro_variables}')
            self.plan_run_tactic(tactic.replace('<token>', self.get_word_at_position(action)))
        return False

    def process_results(self):
        if not self.used_masked_action:
            self.update_lean_infos(self.lean.get_result(self.agent_id))
            self.update_goals()
            self.check_done()
            self.update_obs()
            self.update_masked_actions()
        return self.obs, self.done, self.finished_proof

    def close(self):
        super().close()
    
    def check_done(self):
        if self.n_goals == 0:
            self.done = self.finished_proof = True
