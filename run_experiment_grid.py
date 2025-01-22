import warnings
import torch
from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train
from env.SmartMeter_env_1 import SmartMeterEnv1

if __name__ == '__main__':
    eg = ExperimentGrid(exp_name='Benchmark_Safety_SmartMeter')

    # Set the algorithms.
    my_policy = ['PPOLag']

    # Set the environments.
    mujoco_envs = [
        'SafetyAntVelocity-v1',
        'SafetyHopperVelocity-v1',
        'SafetyHumanoidVelocity-v1',
        'SafetyWalker2dVelocity-v1',
        'SafetyHalfCheetahVelocity-v1',
        'SafetySwimmerVelocity-v1',
    ]

    my_envs = ['SmartMeter1-v0']
    eg.add('env_id', my_envs)

    # Set the device.
    avaliable_gpus = list(range(torch.cuda.device_count()))
    gpu_id = [4]
    # if you want to use CPU, please set gpu_id = None
    # gpu_id = None

    if gpu_id and not set(gpu_id).issubset(avaliable_gpus):
        warnings.warn('The GPU ID is not available, use CPU instead.', stacklevel=1)
        gpu_id = None

    eg.add('algo', my_policy)
    eg.add('logger_cfgs:use_wandb', [False])
    eg.add('logger_cfgs:save_model_freq', [10])
    # eg.add('train_cfgs:vector_env_nums', [4])
    # eg.add('train_cfgs:torch_threads', [1])
    eg.add('algo_cfgs:steps_per_epoch', [720])
    eg.add('algo_cfgs:update_iters', [15])
    eg.add('algo_cfgs:gamma', [1.0])
    eg.add('algo_cfgs:cost_gamma', [1.0])
    eg.add('lagrange_cfgs:cost_limit', [0, 5, 10, 15, 20, 25, 50, 100])
    eg.add('lagrange_cfgs:lagrangian_multiplier_init', [0.001])
    eg.add('lagrange_cfgs:lambda_lr', [0.05])
    eg.add('train_cfgs:total_steps', [1440000])
    eg.add('seed', [0])
    # total experiment num must can be divided by num_pool
    # meanwhile, users should decide this value according to their machine
    eg.run(train, num_pool=12, gpu_id=gpu_id)

    # just fill in the name of the parameter of which value you want to compare.
    # then you can specify the value of the parameter you want to compare,
    # or you can just specify how many values you want to compare in single graph at most,
    # and the function will automatically generate all possible combinations of the graph.
    # but the two mode can not be used at the same time.
    eg.analyze(parameter='lagrange_cfgs:cost_limit', values=[0, 5, 10, 15, 20, 25, 50, 100], compare_num=None, cost_limit=None)
    # eg.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
    # eg.evaluate(num_episodes=1)