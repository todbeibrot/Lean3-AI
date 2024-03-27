from envs.solver_train import SolverTrainEnv
from envs.solver import Solver
from ray.rllib.env.env_context import EnvContext
from gymnasium.spaces import Dict, Box, Discrete

class RaySolverTrainEnv(SolverTrainEnv):


    def __init__(self, config: EnvContext):
        env_kws = [
            'dir_path',
            'use_wandb_logging',
            #'wandb_run',
            'lean_reset_steps',
            'image_len',
            'image_lines',
            'lemmagen_max_steps',
            'solver_max_steps',
            'max_variables',
            'max_intro_variables',
            'max_goals',
            'max_number',
            'verbose',
            'n_solvers',
            'generate_lemma_reward',
            'n_envs',
            'run_name',
            'project_name',
            'continue_training',
            'solve_sorrys'
        ]
        env_kwargs = {}
        for kw in env_kws:
            if kw in config:
                env_kwargs[kw] = config[kw]
        if isinstance(config, EnvContext):
            env_kwargs['id'] = config.worker_index
        super().__init__(**env_kwargs)



class RaySolverEnv(Solver):


    def __init__(self, config: EnvContext):
        env_kws = [
            'file_path',
            'dir_path',
            'image_len',
            'image_lines5',
            'max_steps',
            'verbose',
            'max_variables',
            'max_intro_variables',
            'max_goals',
            'max_number'
        ]
        env_kwargs = {}
        for kw in env_kws:
            if kw in config:
                env_kwargs[kw] = config[kw]
        super().__init__(**env_kwargs)
