from os.path import join, exists, isfile
import time
from random import randint, choice
import json
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union

class LeanGym():


    def __init__(
        self,
        dir_path,
        decl: str,
        verbose: int = 0,
        solve_sorry: bool = False,
        n_solvers: int = 1,
        thm_path: str = ''
    ):
        self.dir_path = dir_path
        self.solve_sorry = solve_sorry
        self.n_solvers = n_solvers
        if solve_sorry:
            self.lean_gym_path = join(dir_path, 'lean-gym-eval')
        else:
            self.lean_gym_path = join(dir_path, 'lean-gym')
        self.decl = decl
        self.verbose = verbose
        self.lean_start_command = ['lean', '--run', 'src/repl.lean']
        self.start()
        self.lemmas = []
        self.theorems = []
        self.tactic_history = []
        self.planned_tactics = {}
        self.results = {}
        if solve_sorry:
            self.thm_load_counter = 0
        else:
            self.load_lemmas()
        if thm_path:
            self.load_theorems(thm_path)
        self.can_load_lemmas = (len(self.lemmas) > 0)
        self.start_tactic = ''

    def close(self):
        self.lean.terminate()
        self.lean.wait(30)
        self.lean.kill()
        self.save_lemmas()
        
    def start(self):
        self.lean = subprocess.Popen(self.lean_start_command, cwd=self.lean_gym_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding='utf-8')
    
    # Reopen lean-gym
    def reset_lean_gym(self):
        self.lean.terminate()
        self.lean.wait(300)
        self.lean.kill()
        self.start()

    # Run a command, and return the result as dict: error, search_id, tactic_state_id, tactic_state, proof_steps
    def run_lean_cmd(self, cmd: str) -> Dict:
        if self.verbose:
            print(cmd)
        # Send the command to lean-gym
        self.lean.stdin.write(cmd)
        self.lean.stdin.flush()
        return json.loads(self.lean.stdout.readline())

    def init_search(self, decl: Optional[str] = None, use_start_tactic: bool = True):
        if decl is None:
            decl = self.decl
        msg = self.run_lean_cmd(f'["init_search", ["{decl}", ""]]\n')
        self.search_id = msg['search_id']
        if use_start_tactic and self.start_tactic:
            if msg['error']:
                print('init error:')
                print(msg['error'])
            else:
                msg = self.run_tac(msg['search_id'], msg['tactic_state_id'], self.start_tactic)
        self.tactic_history = []
        return msg

    def plan_run_tactic(self, search_id: int, tactic_state_id: int, tactic: str, agent_id: int):
        self.planned_tactics[agent_id] = (search_id, tactic_state_id, tactic)
   
    def run_planned_tactics(self):
        self.results = {}
        for search_id, tactic_state_id, tactic in self.planned_tactics.values():
            self.lean.stdin.write(f'["run_tac",["{search_id}","{tactic_state_id}","{tactic}"]]\n')
        self.lean.stdin.flush()
        for agent_id in self.planned_tactics.keys():
            self.results[agent_id] = json.loads(self.lean.stdout.readline())
        self.planned_tactics = {}

    def get_result(self, agent_id):
        return self.results[agent_id]

    def run_tac(self, search_id: int, tactic_state_id: int, tactic: str):
        msg = self.run_lean_cmd(f'["run_tac",["{search_id}","{tactic_state_id}","{tactic}"]]\n')
        if msg['error'] is None:
            self.tactic_history.append(tactic)
        return msg

    def clear_search(self, search_id: Optional[int] = None):
        if search_id is None:
            search_id = self.search_id
        return self.run_lean_cmd(f'["clear_search",["{search_id}"]]\n')

    def shrink_proof(self, search_id, tactic_state_id):
        return self.run_lean_cmd(f'["shrink_proof",["{search_id}","{tactic_state_id}"]]\n')
    
    def load_lemma(self):
        lemma = choice(self.lemmas)
        msg = self.init_search()
        msg = self.run_tac(msg['search_id'], msg['tactic_state_id'], lemma)
        if msg['error']:
            print('load lemma error:')
            print(msg['error'])
            print(lemma)
            print(msg['tactic_state'])
            self.lemmas.remove(lemma)
            return self.load_lemma()
        return msg['search_id'], msg['tactic_state_id'], msg['tactic_state']
    
    def load_theorem(self):
        if self.solve_sorry:
            theorem = self.theorems[self.thm_load_counter % len(self.theorems)]
            self.thm_load_counter += 1
        else:
            theorem = choice(self.theorems)
        msg = self.init_search(theorem, use_start_tactic=False)
        if msg['error'] and self.theorems:
            # print(f'load theorem {theorem} failed')
            # print(msg['error'])
            self.theorems.remove(theorem)
            return self.load_theorem()
        new_msg = msg
        i = 0
        while not new_msg['error'] and i < 50:
            msg = new_msg
            new_msg = self.run_tac(new_msg['search_id'], new_msg['tactic_state_id'], tactic=f'intro h{i}')
        return msg['search_id'], msg['tactic_state_id'], msg['tactic_state']

    def save_lemma(self):
        self.lemmas.append(','.join(self.tactic_history))
        self.can_load_lemmas = True

    def save_lemmas(self):
        with open(join(self.dir_path, 'data', 'lemmas', f'lemmas_{time.time()}_{randint(0, 2**15)}.txt'), 'w', encoding='utf-8') as file:
            for lemma in self.lemmas:
                file.write(lemma + '\n')

    def load_lemmas(self):
        file_path = join(self.dir_path, 'data', 'lemmas', 'all.lean')
        if exists(file_path) and isfile(file_path):
            self.lemmas = open(file_path, 'r', encoding='utf-8').readlines()
        for lemma in range(len(self.lemmas)):
            self.lemmas[lemma] = self.lemmas[lemma].removesuffix('\n')

    def load_theorems(self, thm_path: str):
        with open(thm_path, 'r', encoding='utf-8') as theorems_file:
            for theorem in theorems_file:
                self.theorems.append(theorem.removesuffix('\n'))

    def set_start_tactic(self, tactic: str):
        self.start_tactic = tactic

    def print_infos(self, msg):
        if msg['error']:
            print('error: ' + msg['error'])
        if msg['search_id']:
            print('search_id: ' + msg['search_id'])
        if msg['tactic_state_id']:
            print('tactic_state_id: ' + msg['tactic_state_id'])
        if msg['tactic_state']:
            print('tactic_state: ' + msg['tactic_state'])
        if msg['proof_steps']:
            print('proof_steps: ' + str(msg['proof_steps']))
