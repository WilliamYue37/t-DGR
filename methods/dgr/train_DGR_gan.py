import argparse
import math
import torch
import socket
import datetime
import os

from mlp import MLP
from trainer import Trainer as LearnerTrainer
from metaworld_dataset import MetaworldDataset, ImageDataset
from wgan_gp import Generator, Discriminator, Trainer as GANTrainer
from unet import TemporalUnet

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs for training the learner per task')
parser.add_argument('--gan_epochs', type=int, default=300, help='number of epochs for training the gan per task')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument('--dim', type=int, default=128, help='dimension of the unet')
parser.add_argument('--learner_ckpt', type=str, default=None, help='path to learner checkpoint (*.pt)')
parser.add_argument('--gen_ckpt', type=str, default=None, help='path to generator checkpoint (*.pt)')
parser.add_argument('--ratio', type=float, default=0.9, help='ratio of generated data replayed to learner (must be in the range [0, 1))')
parser.add_argument('--ckpt_folder', type=str, default=None, help='folder to save checkpoints and logs')
parser.add_argument('--warmup', type=int, default=600, help='number of training epochs to warmup the generator')
parser.add_argument('--dataset', type=str, required=True, help='path to dataset of expert demonstrations')
parser.add_argument('--benchmark', type=str, choices=['cw20', 'cw10', 'gcl'], default='cw20', help='benchmark to run')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loading')
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
if args.benchmark == 'cw20' or args.benchmark == 'cw10':
    env_names = ['hammer-v2', 'push-wall-v2', 'faucet-close-v2', 'push-back-v2', 'stick-pull-v2', 'handle-press-side-v2', 'push-v2', 'shelf-place-v2', 'window-close-v2', 'peg-unplug-side-v2']
else:
    env_names = ['bucket0', 'bucket1', 'bucket2', 'bucket3', 'bucket4', 'bucket5', 'bucket6', 'bucket7', 'bucket8', 'bucket9']
repeats = 2 if args.benchmark == 'cw20' else 1

learner_model = MLP(input=49, output=4).cuda()
generator_model = Generator(args.latent_dim, 10, 39).cuda()
discriminator_model = Discriminator(39, 10).cuda()

learner_trainer = None
generator_trainer = None

# load checkpoints
if args.learner_ckpt is not None:
    learner_trainer = LearnerTrainer(learner_model, MetaworldDataset(f'{args.dataset}/{env_names[0]}'), ckpts_folder=args.ckpt_folder, train_batch_size=args.batch_size, train_lr=args.lr, num_workers=args.num_workers) 
    learner_trainer.load(args.learner_ckpt)
if args.gen_ckpt is not None:
    generator_trainer = GANTrainer(
        generator_model,
        discriminator_model,
        ImageDataset(f'{args.dataset}/{env_names[0]}'),
        train_batch_size = args.batch_size,
        ckpts_folder = args.ckpt_folder,
        num_workers = args.num_workers
    )
    generator_trainer.load(args.gen_ckpt)

# continual learning
maxi = torch.load(args.dataset + '/maxi.pt')
for repeat in range(repeats):
    for env_name in env_names:
        print(f'Running {env_name} - {repeat}')
        # get dataset
        learner_dataset = MetaworldDataset(f'{args.dataset}/{env_name}')
        generator_dataset = ImageDataset(f'{args.dataset}/{env_name}')

        # add fake generated data
        if learner_trainer is not None and generator_trainer is not None:
            prev_learner = learner_trainer.model
            prev_generator = generator_trainer.generator

            num_generated_samples = args.ratio * len(learner_dataset) / (1 - args.ratio)
            batches_needed = math.ceil(num_generated_samples / args.batch_size)
            if 'cw' in args.benchmark:
                num_of_env_so_far = env_names.index(env_name) if repeat == 0 else len(env_names)
            elif args.benchmark == 'gcl':
                num_of_env_so_far = min(env_names.index(env_name) + 1, len(env_names))
            else:
                assert False

            for j in range(max(batches_needed, num_of_env_so_far)):
                task_id = j % num_of_env_so_far
                cond = torch.eye(len(env_names))[task_id].cuda()
                cond = torch.broadcast_to(cond, (args.batch_size, len(env_names)))
                traj_data = prev_generator.sample(cond)
                
                with torch.no_grad():
                    target = prev_learner(traj_data).numpy(force=True)
                    data = traj_data.cpu()

                    assert data.shape[0] == args.batch_size
                    for i in range(data.shape[0]):
                        learner_dataset.add_item([data[i], target[i]])
                        generator_dataset.add_item(data[i])
            
        if learner_trainer is None or generator_trainer is None: # initialize trainers for the first time
            learner_trainer = LearnerTrainer(learner_model, learner_dataset, ckpts_folder=args.ckpt_folder, train_batch_size=args.batch_size, train_lr=args.lr, num_workers=args.num_workers)
            generator_trainer = GANTrainer(
                generator_model,
                discriminator_model,
                generator_dataset,
                train_batch_size = args.batch_size,
                ckpts_folder = args.ckpt_folder,
                num_workers = args.num_workers
            )
        else:
            learner_trainer.load_new_dataset(learner_dataset)
            generator_trainer.load_new_dataset(generator_dataset)

        warmup_epochs = args.warmup if env_names.index(env_name) == 0 else 0
        if 'cw' in args.benchmark:
            num_of_env_so_far = env_names.index(env_name) if repeat == 0 else len(env_names)
        elif args.benchmark == 'gcl':
            num_of_env_so_far = min(env_names.index(env_name) + 1, len(env_names))

        learner_trainer.train(args.epochs)
        generator_trainer.train(args.gan_epochs + warmup_epochs)

        learner_trainer.save(env_name + f'-{repeat}')

