import argparse
import math
import threading
import torch
import socket
import datetime
import os

from mlp import MLP, TimeMLP
from trainer import LearnerTrainer, PredictorTrainer
from metaworld_dataset import PolicyDataset, DynamicsDataset, StartStateDataset
from diffusion import GaussianDiffusion, Trainer as DiffusionTrainer


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs for training the learner per task')
parser.add_argument('--steps', type=int, default=10000, help='number of training steps for diffusion per task')
parser.add_argument('--timesteps', type=int, default=1000, help='number of timesteps in the diffusion process')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--learner_ckpt', type=str, default=None, help='path to the learner checkpoint (*.pt)')
parser.add_argument('--gen_ckpt', type=str, default=None, help='path to the generator checkpoint (*.pt)')
parser.add_argument('--ratio', type=float, default=0.85, help='ratio of generated data to real data (must be between [0, 1))')
parser.add_argument('--ckpt_folder', type=str, default=None, help='folder to save checkpoints and logs')
parser.add_argument('--warmup', type=int, default=50000, help='number of training steps to warmup the generator')
parser.add_argument('--dataset', type=str, required=True, help='path to the dataset')
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
generator_model = TimeMLP(input=49, output=39).cuda()
diffusion = GaussianDiffusion(
    generator_model,
    obs_dim = 39,
    cond_dim = 10,
    timesteps = args.timesteps,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()
predictor_model = MLP(input = 49 + 4, output=39).cuda()

learner_trainer = None
generator_trainer = None
predictor_trainer = None

# continual learning
for repeat in range(repeats):
    for env_name in env_names:
        # get dataset
        learner_dataset = PolicyDataset(f'{args.dataset}/{env_name}')
        generator_dataset = StartStateDataset(f'{args.dataset}/{env_name}')
        predictor_dataset = DynamicsDataset(f'{args.dataset}/{env_name}')

        # add fake generated data
        if learner_trainer is not None and generator_trainer is not None and predictor_trainer is not None:
            prev_learner = learner_trainer.model
            prev_generator = generator_trainer.ema_model
            prev_predictor = predictor_trainer.model

            with torch.no_grad():
                max_length = 200
                num_generated_samples = args.ratio * len(learner_dataset) / (1 - args.ratio)
                num_generated_trajs = math.ceil(num_generated_samples / args.batch_size / max_length)
                if args.benchmark == 'cw20':
                    num_of_env_so_far = env_names.index(env_name) if repeat == 0 else len(env_names)
                else:
                    num_of_env_so_far = min(env_names.index(env_name) + 1, len(env_names))

                for j in range(num_generated_trajs):
                    task_id = j % num_of_env_so_far
                    task_cond = torch.eye(10)[task_id].cuda()

                    start_state = prev_generator.sample(task_cond=task_cond, batch_size=args.batch_size)
                    
                    for i in range(max_length):
                        assert start_state.shape == (args.batch_size, 49)
                        action = prev_learner(start_state)
                        next_state = prev_predictor(torch.cat([start_state, action], dim=1))
                        next_state = torch.cat((next_state, task_cond.repeat(args.batch_size, 1)), dim=1)
                        
                        start_state = start_state.cpu()
                        action = action.cpu()
                        next_state_cpu = next_state.clone().cpu()
                        for k in range(start_state.shape[0]):
                            if i == 0: generator_dataset.add_item(start_state[k])
                            learner_dataset.add_item([start_state[k], action[k].numpy(force=True)])
                            predictor_dataset.add_item([start_state[k], action[k], next_state_cpu[k]])

                        start_state = next_state
            
        if learner_trainer is None or generator_trainer is None or predictor_trainer is None: # initialize trainers for the first time
            learner_trainer = LearnerTrainer(learner_model, learner_dataset, ckpts_folder=args.ckpt_folder, train_batch_size=args.batch_size, train_lr=args.lr)
            generator_trainer = DiffusionTrainer(
                diffusion,
                generator_dataset,
                train_batch_size = args.batch_size,
                train_lr = args.lr,
                save_every = 100,
                gradient_accumulate_every = 2,    # gradient accumulation steps
                ema_decay = 0.995,                # exponential moving average decay
                amp = True,                        # turn on mixed precision
                results_folder = args.ckpt_folder
            )
            predictor_trainer = PredictorTrainer(predictor_model, predictor_dataset, cond_dim=10, ckpts_folder=args.ckpt_folder, train_batch_size=args.batch_size, train_lr=args.lr)
        else:
            learner_trainer.load_new_dataset(learner_dataset)
            generator_trainer.load_new_dataset(generator_dataset)
            predictor_trainer.load_new_dataset(predictor_dataset)

        warmup_steps = args.warmup if env_names.index(env_name) == 0 else 0

        learn_thread = threading.Thread(target=learner_trainer.train, args=(args.epochs,))
        generator_thread = threading.Thread(target=generator_trainer.train, args=(args.steps + warmup_steps,))
        predictor_thread = threading.Thread(target=predictor_trainer.train, args=(args.epochs,))
        learn_thread.start()
        generator_thread.start()
        predictor_thread.start()
        learn_thread.join()
        generator_thread.join()
        predictor_thread.join()

        learner_trainer.save(env_name + f'-{repeat}')
    # generator_trainer.save(env_name)
    # predictor_trainer.save(env_name)

