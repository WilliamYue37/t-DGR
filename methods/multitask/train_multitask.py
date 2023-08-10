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
parser.add_argument('--epochs', type=int, default=500, help='number of epochs for training the learner per task')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--ckpt', type=str, default=None, help='path to the learner checkpoint (*.pt)')
parser.add_argument('--dataset', type=str, required=True, help='path to dataset of expert demonstrations')
parser.add_argument('--ckpt_folder', type=str, default=None, help='folder to save checkpoints and logs')
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

# get dataset
dataset = MetaworldDataset(args.dataset)

# define model
model = MLP(input=49, output=4).cuda()
trainer = Trainer(model, dataset, train_batch_size=args.batch_size, ckpts_folder=args.ckpt_folder, train_lr=args.lr) 

# load checkpoint
if args.ckpt:
    trainer.load(args.ckpt)

trainer.train(args.epochs)
trainer.save('multitask')

