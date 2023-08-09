import torch
import numpy as np
import os
from mlp import MLP
import argparse
from trainer import Trainer
from metaworld_dataset import MetaworldDataset

from tests.metaworld.envs.mujoco.sawyer_xyz.test_scripted_policies import ALL_ENVS

env_names = ['hammer-v2', 'push-wall-v2', 'faucet-close-v2', 'push-back-v2', 'stick-pull-v2', 'handle-press-side-v2', 'push-v2', 'shelf-place-v2', 'window-close-v2', 'peg-unplug-side-v2']

parser = argparse.ArgumentParser()
parser.add_argument('--models', type=str, default=None, help='path to the directory containing the learner checkpoints')
parser.add_argument('--runs', type=int, default=10, help='number of runs to average over')
parser.add_argument('--benchmark', type=str, choices=['cw20', 'gcl', 'cw10'], default='cw20', help='benchmark to evaluate on')
args = parser.parse_args()

# set benchmark specific settings
if 'cw' in args.benchmark:
    model_names = ['hammer-v2', 'push-wall-v2', 'faucet-close-v2', 'push-back-v2', 'stick-pull-v2', 'handle-press-side-v2', 'push-v2', 'shelf-place-v2', 'window-close-v2', 'peg-unplug-side-v2']
else:
    model_names = ['bucket0', 'bucket1', 'bucket2', 'bucket3', 'bucket4', 'bucket5', 'bucket6', 'bucket7', 'bucket8', 'bucket9']
repeats = 2 if args.benchmark == 'cw20' else 1
dataset = MetaworldDataset('/scratch/cluster/william/metaworld/state_data/')

final_model_path = args.models + '/model-' + model_names[len(model_names) - 1] + f'-{repeats - 1}.pt'
model = MLP(input=49, output=4).cuda()
final_trainer = Trainer(model, dataset, train_batch_size=32, ckpts_folder='trash', train_lr=0.0001)
final_trainer.load(final_model_path)

def compute_task_success(env_name, trainer, repeat = 0):
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
                a = trainer.eval(9, obs).squeeze(0).cpu().numpy()
                obs, reward, done, info = env.step(a)

                if info['success']:
                    successes += 1
                    break
                
                steps += 1
    return successes / args.runs

def compute_success():
    '''Computes the average number of successes across all tasks'''
    total_successes = 0
    for repeat in range(repeats):
        for env_name in env_names:
            total_successes += compute_task_success(env_name, final_trainer, repeat=repeat)
    
    return total_successes / len(env_names) / repeats

def compute_forgetting():
    total_forgetting = 0
    for repeat in range(repeats):
        for env_name in env_names:
            model_name = args.models + '/model-' + env_name + f'-{repeat}.pt'
            model = MLP(input=49, output=4).cuda()
            trainer = Trainer(model, dataset, train_batch_size=32, ckpts_folder='trash', train_lr=0.0001)
            trainer.load(model_name)
            success_rate_i = compute_task_success(env_name, trainer, repeat=repeat)
            success_rate_f = compute_task_success(env_name, final_trainer, repeat=repeat)
            total_forgetting += success_rate_i - success_rate_f
    
    return total_forgetting / len(env_names) / repeats

def compute_FT():
    '''compute the average forward transfer across all tasks'''
    total_FT = 0
    for repeat in range(repeats):
        for i in range(len(env_names)):
            if i == 0 and repeat == 0:
                success_rate_prev = 0
            else:
                model_name = args.models + '/model-' + env_names[i - 1] + f'-{repeat}.pt'
                model = MLP(input=49, output=4).cuda()
                trainer = Trainer(model, dataset, train_batch_size=32, ckpts_folder='trash', train_lr=0.0001)
                trainer.load(model_name)
                prev_repeat = 0 if repeat == 0 or i == 0 else 1
                success_rate_prev = compute_task_success(env_names[i], trainer, repeat=prev_repeat)

            model_name = args.models + '/model-' + env_names[i] + f'-{repeat}.pt'
            model = MLP(input=49, output=4).cuda()
            trainer = Trainer(model, dataset, train_batch_size=32, ckpts_folder='trash', train_lr=0.0001)
            trainer.load(model_name)
            success_rate_cur = compute_task_success(env_names[i], trainer, repeat=repeat)

            AUC = (success_rate_cur + success_rate_prev) / 2
            total_FT += (AUC - 0.5) / 0.5

    return total_FT / len(env_names) / repeats


print('Avg Successes per Env:', compute_success())
if args.benchmark == 'cw20': print('Avg Forgetting per Env:', compute_forgetting())
if args.benchmark == 'cw20': print('Avg Forward Transfer per Env:', compute_FT())






