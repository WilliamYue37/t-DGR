import argparse
import socket
import datetime
import os
import torch

from mlp import MLP
from trainer import Trainer as LearnerTrainer
from metaworld_dataset import MetaworldDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
parser.add_argument('--epochs', type=int, default=250, help='number of epochs for training the learner per task')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--ckpt', type=str, default=None, help='path to the learner checkpoint (*.pt)')
parser.add_argument('--ckpt_folder', type=str, default=None, help='folder to save checkpoints and logs')
parser.add_argument('--ewc_lambda', type=float, default=100, help='fisher multiplier for EWC')
parser.add_argument('--dataset', type=str, required=True, help='path to dataset of expert demonstrations')
parser.add_argument('--benchmark', type=str, choices=['cw20', 'gcl', 'cw10'], default='cw20', help='benchmark to run')
parser.add_argument('--seed', type=int, default=0, help='random seed')
args = parser.parse_args()

torch.manual_seed(args.seed)

# create runs folder
if not os.path.exists('runs'):
    os.mkdir('runs')

# create ckpts folder
if args.ckpt_folder is None:
    args.ckpt_folder = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + f'-{socket.gethostname()}'
args.ckpt_folder = 'runs/' + args.ckpt_folder

# set benchmark specific settings
if 'cw' in args.benchmark:
    env_names = ['hammer-v2', 'push-wall-v2', 'faucet-close-v2', 'push-back-v2', 'stick-pull-v2', 'handle-press-side-v2', 'push-v2', 'shelf-place-v2', 'window-close-v2', 'peg-unplug-side-v2']
else:
    env_names = ['bucket0', 'bucket1', 'bucket2', 'bucket3', 'bucket4', 'bucket5', 'bucket6', 'bucket7', 'bucket8', 'bucket9']
repeats = 2 if args.benchmark == 'cw20' else 1

learner_model = MLP(input=49, output=4).cuda()

learner_trainer = None

# load checkpoints
if args.ckpt is not None:
    learner_trainer = LearnerTrainer(learner_model, MetaworldDataset(f'{args.dataset}/{env_names[0]}'), ckpts_folder=args.ckpt_folder, train_batch_size=args.batch_size, train_lr=args.lr, ckpt_every=500) 
    learner_trainer.load(args.ckpt)

# continual learning
for repeat in range(repeats):
    for task_id, env_name in enumerate(env_names):
        # get dataset
        learner_dataset = MetaworldDataset(f'{args.dataset}/{env_name}')
            
        if learner_trainer is None: # initialize trainers for the first time
            learner_trainer = LearnerTrainer(learner_model, learner_dataset, ckpts_folder=args.ckpt_folder, train_batch_size=args.batch_size, train_lr=args.lr)
        else:
            learner_trainer.load_new_dataset(learner_dataset)

        learner_trainer.train(args.epochs)

        learner_trainer.save(env_name + f'-{repeat}')

