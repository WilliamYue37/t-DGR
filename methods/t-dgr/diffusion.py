import copy
import torch
import os

import torch.nn.functional as F
from einops import rearrange
from einops_exts import check_shape
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils import data
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):
    while True:
        for (data, traj_time) in dl:
            yield data, traj_time

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        horizon,
        obs_dim,
        cond_dim,
        timesteps = 1000,
        loss_type = 'l1',
        use_dynamic_thres = False, # from the Imagen paper
        dynamic_thres_percentile = 0.9
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.cond_dim = cond_dim 

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # register buffer helper function that casts float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, traj_time, task_cond, clip_denoised: bool):
        x = torch.cat((x, task_cond), dim=2) # add conditioning frame
        pred_noise = self.denoise_fn(x, t, traj_time)
        pred_noise = pred_noise[:, :, :self.obs_dim] # remove conditioning frame
        x = x[:, :, :self.obs_dim] # remove conditioning frame
        x_recon = self.predict_start_from_noise(x, t=t, noise = pred_noise)

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim = -1
                )

                s.clamp_(min = 1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x, t, traj_time, task_cond, clip_denoised = True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x, t, traj_time, task_cond, clip_denoised = clip_denoised)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, shape, traj_time, task_cond = None):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), traj_time, task_cond = task_cond)

        # add conditioning frame back in
        img = torch.cat((img, task_cond), dim=2)
        return img

    @torch.inference_mode()
    def sample(self, traj_time, task_cond, batch_size = 16):
        task_cond = torch.broadcast_to(task_cond, (batch_size, self.horizon, self.cond_dim)).cuda()
        return self.p_sample_loop((batch_size, self.horizon, self.obs_dim), traj_time, task_cond = task_cond)

    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, diffusion_time, traj_time, task_cond, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=diffusion_time, noise=noise)
        x_noisy = torch.cat((x_noisy, task_cond), dim=2) # add one-hot task vector to "t" dimension

        x_recon = self.denoise_fn(x_noisy, diffusion_time, traj_time)
        x_recon = x_recon[:, :, :self.obs_dim] # remove conditioning frame

        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, traj_time):
        # extract conditional one-hot task vector
        task_cond = x[:, :, self.obs_dim:]
        x = x[:, :, :self.obs_dim]

        b, device = x.shape[0], x.device
        check_shape(x, 'b h t', h = self.horizon, t = self.obs_dim)
        check_shape(task_cond, 'b h c', h = self.horizon, c = self.cond_dim)
        check_shape(traj_time, 'b')
        diffusion_time = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, diffusion_time, traj_time, task_cond)
    
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        *,
        ema_decay = 0.995,
        train_batch_size = 32,
        train_lr = 1e-4,
        gradient_accumulate_every = 2,
        amp = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_every = 1000,
        results_folder = './results',
        max_grad_norm = None,
        num_workers = 4
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_every = save_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.ds = dataset

        self.num_workers = num_workers
        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True, num_workers=num_workers))
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled = amp)
        self.max_grad_norm = max_grad_norm

        self.writer = SummaryWriter(log_dir=results_folder + '/generator_logs')
        self.results_folder = Path(results_folder + '/generator_ckpts')
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

        # delete previous milestone
        if isinstance(milestone, int):
            last_mile = str(self.results_folder / f'model-{milestone - 1}.pt')
            if os.path.exists(last_mile):
                os.remove(last_mile)

    def load(self, milestone, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1]) for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        self.scaler.load_state_dict(data['scaler'])

    def load_new_dataset(self, dataset):
        self.ds = dataset
        self.dl = cycle(data.DataLoader(self.ds, batch_size = self.batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers))

    def train(self, train_num_steps = 100000):
        for _ in range(train_num_steps):
            for i in range(self.gradient_accumulate_every):
                data, traj_time = next(self.dl)
                data, traj_time = data.cuda(), traj_time.squeeze(dim=1).cuda()

                with autocast(enabled = self.amp):
                    loss = self.model(data, traj_time)

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                if self.step % 100 == 0: self.writer.add_scalar('Loss/train', loss.item(), self.step)

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_every == 0:
                milestone = self.step // self.save_every
                self.save(milestone)
            
            self.step += 1

