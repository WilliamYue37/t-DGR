import argparse
import torch
from mlp import MLP
from trainer import Trainer
from metaworld_dataset import MetaworldDataset
import socket
import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument('--ckpt_folder', type=str, default=None)
parser.add_argument('--dataset', type=str, default='/scratch/cluster/william/metaworld/state_data')
parser.add_argument('--benchmark', type=str, choices=['cw20', 'gcl'], default='cw20')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)

# create runs folder
if not os.path.exists('runs'):
    os.mkdir('runs')

# create ckpts folder
if args.ckpt_folder is None:
    args.ckpt_folder = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + f'-{socket.gethostname()}'
args.ckpt_folder = 'runs/' + args.ckpt_folder

# get dataset
dataset = MetaworldDataset(args.dataset)

# define model
model = MLP(input=49, output=4).cuda()

# set benchmark specific settings
if args.benchmark == 'cw20':
    env_names = ['hammer-v2', 'push-wall-v2', 'faucet-close-v2', 'push-back-v2', 'stick-pull-v2', 'handle-press-side-v2', 'push-v2', 'shelf-place-v2', 'window-close-v2', 'peg-unplug-side-v2']
else:
    env_names = ['bucket0', 'bucket1', 'bucket2', 'bucket3', 'bucket4', 'bucket5', 'bucket6', 'bucket7', 'bucket8', 'bucket9']
repeats = 2 if args.benchmark == 'cw20' else 1

trainer = None

# continual learning
for repeat in range(repeats):
    for env_name in env_names:
        # get dataset
        dataset = MetaworldDataset(f'{args.dataset}/{env_name}')

        if trainer is None:
            trainer = Trainer(model, dataset, train_batch_size=args.batch_size, ckpts_folder=args.ckpt_folder, train_lr=args.lr) 
        else:
            trainer.load_new_dataset(dataset)

        trainer.train(args.epochs)
        trainer.save(env_name + f'-{repeat}')