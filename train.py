import pathlib
import wandb
from os.path import join
from utils.utils import get_latest_run_id
from utils.scripts import merge_corpi, merge_lemmas, build_word2vec
from envs.ray_env import RaySolverTrainEnv
from ray.rllib.algorithms.appo import APPOConfig
from ray.tune.registry import register_env


if __name__ == "__main__":
    verbose = 0
    continue_training = True
    merge_all_lemmas = False
    update_word2vec = False
    use_wandb = True
    num_worker = 7
    n_solver = 8
    use_tuner = False

    # Paths and Logging
    file_path = pathlib.Path(__file__)
    dir_path = file_path.parent.resolve()
    data_path = join(dir_path, 'data')
    pathlib.Path(data_path).mkdir(exist_ok=True)
    log_folder = join(dir_path, 'runs')
    run_id = get_latest_run_id(log_folder) + 1 - continue_training
    run_name = f'solver_train_{run_id}'
    project_name = 'solver6_run1'
    run_path = join(log_folder, run_name)
    pathlib.Path(run_path).mkdir(parents=True, exist_ok=True)
    tensorboard_log_path = join(run_path, 'tb_logs')
    monitor_log_path = join(run_path, 'monitor')
    pathlib.Path(monitor_log_path).mkdir(exist_ok=True)
    model_path = join(run_path, 'models')
    pathlib.Path(model_path).mkdir(exist_ok=True)
    model_save_path = join(model_path, 'model')
    if continue_training:
        model_load_path = join(log_folder, f'solver_train_{run_id}', 'models')
        latest_model_id = get_latest_run_id(model_load_path, 'checkpoint_')
        # model_load_path = join(model_load_path, 'checkpoint_' + str(latest_model_id))
        model_load_path = join(model_load_path, 'checkpoint_004238')

    log_interval = 1
    print(f'run name: {run_name}')

    # Scripts
    if merge_all_lemmas:
        merge_lemmas(join(data_path, 'lemmas'), remove_files=True)
        merge_lemmas(join(data_path, 'lemmas_with_proofs'), remove_files=True)
        merge_lemmas(join(data_path, 'solved_lemmas'), remove_files=True)
    if update_word2vec and not continue_training:
        print('start building word2vec model')
        corpi_path = join(data_path, 'corpi')
        merge_corpi(corpi_path, remove_files=True)
        build_word2vec(join(corpi_path, 'corpus.txt'), model_save_path=join(data_path, 'word2vec'), workers=10, min_count=1, vector_size=15, epochs=100)
        print('finished building word2vec model')

    # WandB
    if use_wandb:
        run = wandb.init(
            name=run_name,
            project=project_name,
            sync_tensorboard=False,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=False,  # optional
            # config=env_kwargs
            resume=continue_training,
            group=run_name
        )
    else:
        run = None

    # Environment
    # Hyperparameter
    generate_lemma_reward = 0.05
    env_kwargs = {
        'dir_path': dir_path,
        'use_wandb_logging': use_wandb,
        'lean_reset_steps': 10001,
        'image_len': 20,
        'image_lines': 30,
        'lemmagen_max_steps': 50,
        'solver_max_steps': 50,
        'max_variables': 10,
        'max_intro_variables': 10,
        'max_goals': 30,
        'max_number': 100,
        'verbose': verbose,
        'n_solvers': n_solver,
        'generate_lemma_reward': generate_lemma_reward,
        'n_envs': num_worker,
        'run_name': run_name,
        'project_name': project_name,
        'continue_training': continue_training
    }

    bad_env = RaySolverTrainEnv(env_kwargs)
    lemmagen_obs_space = bad_env._obs_space_in_preferred_format['lemmagen']
    solver_obs_space = bad_env._obs_space_in_preferred_format['solver0']
    lemmagen_action_space = bad_env._action_space_in_preferred_format['lemmagen']
    solver_action_space = bad_env._action_space_in_preferred_format['solver0']
    bad_env.close()

    register_env('solver_train', lambda config: RaySolverTrainEnv(config))

    def one_solver(agent_id: str, episode, worker, **kwargs):
        if agent_id == 'lemmagen':
            return 'lemmagen'
        if agent_id.startswith('solver'):
            return 'solver'

    def multiple_solvers(agent_id: str, episode, worker, **kwargs):
        return agent_id

    solver = {
                'solver': (
                    None,
                    solver_obs_space,
                    solver_action_space,
                    {
                        'model': {
                            'conv_filters': [[16, [4, 4], 2], [32, [4, 4], 2], [64, [3, 3], 2]],
                            'post_fcnet_hiddens': [256] * 3,
                        },
                    }
                )
            }

    solvers = {
                f'solver{agent}': (
                    None,
                    solver_obs_space,
                    solver_action_space,
                    {
                        'model': {
                            'conv_filters': [[16, [4, 4], 2], [32, [4, 4], 2], [64, [3, 3], 2]],
                            'post_fcnet_hiddens': [256] * 3,
                        },
                    }
                )
                for agent in range(n_solver)
            }

    policies_to_train_one_solver = ['lemmagen', 'solver']
    policies_to_train_multiple_solver = ['lemmagen'] + [f'solver{agent}' for agent in range(n_solver)]

    config = (
        APPOConfig()
        .environment('solver_train', env_config=env_kwargs)
        .rollouts(
            num_rollout_workers=num_worker,
            num_envs_per_worker=1
        )
        .resources(
            num_gpus=1,
            num_cpus_per_worker=1,
            num_gpus_per_worker=0,
        )
        .framework('torch')
        .training()
        .multi_agent(
            policies={
                'lemmagen': (
                    None,
                    lemmagen_obs_space,
                    lemmagen_action_space,
                    {
                        'model': {
                            'conv_filters': [[16, [4, 4], 2], [32, [4, 4], 2], [64, [3, 3], 3]],
                            'post_fcnet_hiddens': [128] * 2,
                        },
                    }
                )
            } | solver,
            policy_mapping_fn=multiple_solvers,
            policies_to_train=policies_to_train_one_solver
        )
        .evaluation(evaluation_interval=None)
    )
    algo = config.build()
    if continue_training:
        algo.restore(model_load_path)

    print('start learning')
    for i in range(1000000):
        if use_wandb:
            run.log({'train_steps': i})
        if i % 30 == 4:
            algo.save(model_path)
        result = algo.train()

    algo.stop()