import argparse
import torch
from mlp import MLP
from trainer import Trainer
from metaworld_dataset import MetaworldDataset
import socket
import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
parser.add_argument('--epochs', type=int, default=250, help='number of epochs for training the learner per task')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--ckpt_folder', type=str, default=None, help='folder to save checkpoints and logs')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--dataset', type=str, required=True, help='path to dataset of expert demonstrations')
parser.add_argument('--benchmark', type=str, choices=['cw20', 'gcl', 'cw10'], default='cw20', help='benchmark to run')
args = parser.parse_args()

torch.manual_seed(args.seed)

# create runs folder
if not os.path.exists('runs'):
    os.mkdir('runs')

# create ckpts folder
if args.ckpt_folder is None:
    args.ckpt_folder = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + f'-{socket.gethostname()}'
args.ckpt_folder = 'runs/' + args.ckpt_folder

# get dataset
dataset = MetaworldDataset(args.dataset)

# define model
model = MLP(input=49, output=4).cuda()

# set benchmark specific settings
if 'cw' in args.benchmark:
    env_names = ['hammer-v2', 'push-wall-v2', 'faucet-close-v2', 'push-back-v2', 'stick-pull-v2', 'handle-press-side-v2', 'push-v2', 'shelf-place-v2', 'window-close-v2', 'peg-unplug-side-v2']
else:
    env_names = ['bucket0', 'bucket1', 'bucket2', 'bucket3', 'bucket4', 'bucket5', 'bucket6', 'bucket7', 'bucket8', 'bucket9']
repeats = 2 if args.benchmark == 'cw20' else 1

trainer = None

# continual learning
for repeat in range(repeats):
    for idx, env_name in enumerate(env_names):
        # get dataset
        dataset = MetaworldDataset(f'{args.dataset}/{env_name}')

        if trainer is None:
            trainer = Trainer(model, dataset, train_batch_size=args.batch_size, ckpts_folder=args.ckpt_folder, train_lr=args.lr) 
        else:
            trainer.load_new_dataset(dataset)

        trainer.train(args.epochs)
        prune_percent = 0.5 if idx == 0 and repeat == 0 else 0.75
        trainer.prune(prune_percent=prune_percent)
        trainer.zero_pruned_weights()
        trainer.train(args.epochs // 2) # finetune after pruning
        trainer.next_task()
        trainer.save(env_name + f'-{repeat}')
