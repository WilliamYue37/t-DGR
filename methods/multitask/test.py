import torch
import numpy as np
import os
from mlp import MLP
import argparse

from tests.metaworld.envs.mujoco.sawyer_xyz.test_scripted_policies import ALL_ENVS

env_names = ['hammer-v2', 'push-wall-v2', 'faucet-close-v2', 'push-back-v2', 'stick-pull-v2', 'handle-press-side-v2', 'push-v2', 'shelf-place-v2', 'window-close-v2', 'peg-unplug-side-v2']

argparser = argparse.ArgumentParser()
argparser.add_argument('--model', type=str, default=None, help='path to the directory containing the learner checkpoint')
argparser.add_argument('--runs', type=int, default=10, help='number of runs to average over')
args = argparser.parse_args()

final_model_path = args.model + '/model-multitask.pt'
final_model = MLP(input=49, output=4).cuda()
final_model.load_state_dict(torch.load(final_model_path)['model'])

def compute_task_success(env_name, model):
    '''Computes the number of successes for a given task'''
    successes = 0
    with torch.no_grad():
        for i in range(args.runs):
            env = ALL_ENVS[env_name]()
            env._partially_observable = False
            env._freeze_rand_vec = False
            env._set_task_called = True
            task_id = env_names.index(env_name)

            obs = env.reset()
            steps = 0
            while steps < 200:
                obs = np.concatenate([obs, np.eye(len(env_names))[task_id]])
                obs = torch.tensor(obs, dtype=torch.float32).cuda()
                a = model(obs).squeeze(0).cpu().numpy()
                obs, reward, done, info = env.step(a)

                if info['success']:
                    successes += 1
                    break
                
                steps += 1
    return successes / args.runs

def compute_success():
    '''Computes the average number of successes across all tasks'''
    total_successes = 0
    for env_name in env_names:
        total_successes += compute_task_success(env_name, final_model)
    
    return total_successes / len(env_names)

print('Avg Successes per Env:', compute_success())






