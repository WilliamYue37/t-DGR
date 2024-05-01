from torch.utils.data import DataLoader
from torch.optim import Adam

import torch.nn as nn
import torch.autograd as autograd
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import os
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Generator(nn.Module):
    def __init__(self, latent_dim, task_cond_dim, time_cond_dim, output_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.traj_time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_cond_dim),
            nn.Linear(time_cond_dim, time_cond_dim * 4),
            nn.Mish(),
            nn.Linear(time_cond_dim * 4, time_cond_dim),
        )

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + task_cond_dim + time_cond_dim, 256, normalize=False),
            *block(256, 512),
            *block(512, 1024),
            *block(1024, 2048),
            nn.Linear(2048, output_dim),
            nn.Tanh()
        )

    def forward(self, z, task_cond, traj_time):
        time_emb = self.traj_time_mlp(traj_time)
        z = torch.cat([z, task_cond, time_emb], dim=1)
        obs = self.model(z)
        return obs
    
    def sample(self, task_cond, traj_time): # sample a batch of observations with task conditioning added
        batch_size = task_cond.shape[0]
        z = torch.randn(batch_size, self.latent_dim).cuda()
        obs = self.forward(z, task_cond, traj_time)
        return torch.cat([obs, task_cond], dim=1)

class Discriminator(nn.Module):
    def __init__(self, input_dim, task_cond_dim, time_cond_dim):
        super(Discriminator, self).__init__()

        self.traj_time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_cond_dim),
            nn.Linear(time_cond_dim, time_cond_dim * 4),
            nn.Mish(),
            nn.Linear(time_cond_dim * 4, time_cond_dim),
        )

        self.model = nn.Sequential(
            nn.Linear(input_dim + task_cond_dim + time_cond_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, obs, task_cond, traj_time):
        time_emb = self.traj_time_mlp(traj_time)
        x = torch.cat([obs, task_cond, time_emb], dim=1)
        validity = self.model(x)
        return validity

class Trainer(object):
    def __init__(self, generator, discriminator, dataset, train_batch_size = 64, n_critic = 5, ckpts_folder = './ckpts', num_workers = 4):
        self.generator = generator
        self.discriminator = discriminator
        self.dataset = dataset
        self.ckpts_folder = ckpts_folder
        self.batch_size = train_batch_size
        self.num_workers = num_workers

        self.n_critic = n_critic # number of discriminator updates per generator update
        self.lambda_gp = 10 # Loss weight for gradient penalty

        self.dl = DataLoader(self.dataset, batch_size = train_batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
        self.optimizer_G = Adam(self.generator.parameters(), lr = 1e-4, betas=(0, 0.9)) # default values from wgan-gp paper
        self.optimizer_D = Adam(self.discriminator.parameters(), lr = 1e-4, betas=(0, 0.9)) # default values from wgan-gp paper

        self.writer = SummaryWriter(log_dir=ckpts_folder + '/generator_logs')
        self.results_folder = Path(ckpts_folder + '/generator_ckpts')
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.epoch = 0

    def save(self, milestone):
        "save model: string is for special milestones, int is for regular milestones"
        data = {
            'epoch': self.epoch,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict()
        }
        os.makedirs(str(self.ckpts_folder), exist_ok=True)
        torch.save(data, str(self.ckpts_folder / f'model-{milestone}.pt'))

        # delete previous milestone
        if isinstance(milestone, int):
            last_mile = str(self.ckpts_folder / f'model-{milestone - 1}.pt')
            if os.path.exists(last_mile):
                os.remove(last_mile)

    def load(self, ckpt):
        data = torch.load(ckpt)
        self.epoch = data['epoch']
        self.generator.load_state_dict(data['generator'])
        self.discriminator.load_state_dict(data['discriminator'])

    def load_new_dataset(self, dataset):
        self.dataset = dataset
        self.dl = DataLoader(self.dataset, batch_size = self.batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers)

    def _compute_gradient_penalty(self, real_samples, fake_samples, task_cond, traj_time):
        """Calculates the gradient penalty loss for WGAN GP"""
        batch_size = real_samples.shape[0]
        # Random weight term for interpolation between real and fake samples
        alpha = torch.randn(batch_size, 1).cuda()
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates, task_cond, traj_time)
        fake = torch.ones(batch_size, 1).cuda()
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train(self, num_epoch):
        self.generator.train()
        self.discriminator.train()
        for _ in range(num_epoch):
            total_generator_loss, total_discriminator_loss = 0, 0
            for i, (real_obs, traj_time) in enumerate(self.dl):
                real_obs, traj_time = real_obs.cuda(), traj_time.squeeze(dim=1).cuda()
                task_cond = real_obs[:, -10:]
                real_obs = real_obs[:, :-10]

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Sample noise as generator input
                batch_size = real_obs.shape[0]
                z = torch.randn(batch_size, self.generator.latent_dim).cuda()

                # Generate a batch of images
                fake_obs = self.generator(z, task_cond, traj_time)

                # Real images
                real_validity = self.discriminator(real_obs, task_cond, traj_time)
                # Fake images
                fake_validity = self.discriminator(fake_obs, task_cond, traj_time)
                # Gradient penalty
                gradient_penalty = self._compute_gradient_penalty(real_obs, fake_obs, task_cond, traj_time)
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty

                d_loss.backward()
                self.optimizer_D.step()
                total_discriminator_loss += d_loss.item()

                self.optimizer_G.zero_grad()

                # Train the generator every n_critic steps
                if i % self.n_critic == 0:

                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Generate a batch of images
                    fake_obs = self.generator(z, task_cond, traj_time)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = self.discriminator(fake_obs, task_cond, traj_time)
                    g_loss = -torch.mean(fake_validity)

                    g_loss.backward()
                    self.optimizer_G.step()
                    total_generator_loss += g_loss.item()

            total_generator_loss /= len(self.dl.dataset) // self.n_critic
            total_discriminator_loss /= len(self.dl.dataset)

            self.writer.add_scalar('Loss/train_generator', total_generator_loss, self.epoch)
            self.writer.add_scalar('Loss/train_discriminator', total_discriminator_loss, self.epoch)

            self.epoch += 1