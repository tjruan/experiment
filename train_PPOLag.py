import omnisafe
import torch
import random
import numpy as np
from env.SmartMeter_env_1 import SmartMeterEnv1

if __name__ == '__main__':
    env_id = 'SmartMeter1-v0'
    custom_cfgs = {
        'seed': 0,
        'train_cfgs': {
            'total_steps': 1440000,
            'device': 'cuda:5',
            #'device': 'cpu',
        },
        'algo_cfgs': {
            'steps_per_epoch': 720,
            'update_iters': 10,
            'gamma': 1.0,
            'cost_gamma': 1.0,
        },
        'logger_cfgs': {
            'save_model_freq': 10,
            'log_dir': "/data/runs",
        },
        'lagrange_cfgs': {
            'cost_limit': 0,
            'lagrangian_multiplier_init': 0.001,
            'lambda_lr': 0.05,
        }
    }

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)

    agent = omnisafe.Agent('PPOLag', env_id, custom_cfgs=custom_cfgs)
    agent.learn()