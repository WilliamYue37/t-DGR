import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import copy
from pathlib import Path
import os

class Trainer():
    def __init__(self, model, dataset, train_batch_size = 32, train_lr = 1e-4, ckpt_every = 1000, ckpts_folder = './ckpts'):
        self.model = model

        self.writer = SummaryWriter(log_dir=ckpts_folder + '/learner_logs')
        self.ckpt_every = ckpt_every
        self.ckpts_folder = Path(ckpts_folder + '/learner_ckpts')

        self.batch_size = train_batch_size

        self.ds = dataset
        self.dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True)

        self.opt = Adam(model.parameters(), lr = train_lr)

        self.epoch = 0

        # packnet stuff
        self.current_task_idx = 0
        self.param_to_task = {} # assigns each parameter to a task id 
        self.fixed_weights = {} # stores the fixed weights of the model at the end of each task
        for name, param in self.model.named_parameters():
            self.param_to_task[name] = torch.zeros_like(param)
            self.fixed_weights[name] = copy.deepcopy(param).detach()
        
    # choose which weights in the layer to prune
    def get_mask(self, layer_name, weights, prune_percent):
        '''Ranks weights by magnitude. Sets all below kth to 0. Returns pruned mask.'''

        # get kth value
        avalible_mask = self.param_to_task[layer_name].ge(self.current_task_idx)
        avalible_weights = weights[avalible_mask].abs()
        cutoff_rank = round(prune_percent * avalible_weights.numel())
        cutoff_value = torch.kthvalue(avalible_weights, cutoff_rank).values.item()

        return weights.abs().le(cutoff_value) * avalible_mask

    def prune(self, prune_percent = 0.5):
        # iterate over all layers in the neural network
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                mask = self.get_mask(name, param, prune_percent)
                assert torch.sum(self.param_to_task[name][mask] != self.current_task_idx).item() == 0 # make sure we aren't pruning weights that have already been fixed
                self.param_to_task[name][mask] = self.current_task_idx + 1 # set the pruned weights to the next task id

    def restore_fixed_weights(self, bias = True):
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                layer_mask = self.param_to_task[name].ne(self.current_task_idx)
                param.data[layer_mask] = copy.deepcopy(self.fixed_weights[name][layer_mask])
            elif 'bias' in name and bias: # 'bias' should be false for the first task
                param.data = copy.deepcopy(self.fixed_weights[name])

    def zero_pruned_weights(self, task_id = None):
        if task_id is None:
            task_id = self.current_task_idx
        for name, param in self.model.named_parameters():
            layer_mask = self.param_to_task[name].gt(task_id)
            if 'weight' in name:
                param.data[layer_mask] = 0
                self.fixed_weights[name][layer_mask] = 0

    def next_task(self):
        # save fixed weights
        for name, param in self.model.named_parameters():
            self.fixed_weights[name] = copy.deepcopy(param).detach()

        self.current_task_idx += 1

    def eval(self, task_id, obs):
        # apply task specific mask and save original weights to restore later
        old_state = copy.deepcopy(self.model.state_dict())
        self.zero_pruned_weights(task_id)

        with torch.no_grad():
            obs = obs.cuda()
            output = self.model(obs)

        # restore original weights
        self.model.load_state_dict(old_state)

        return output

    def save(self, milestone):
        "save model: string is for special milestones, int is for regular milestones"
        data = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'param_to_task': self.param_to_task,
            'current_task_idx': self.current_task_idx
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
        self.model.load_state_dict(data['model'])
        self.param_to_task = data['param_to_task']
        self.current_task_idx = data['current_task_idx']

    def load_new_dataset(self, dataset):
        self.ds = dataset
        self.dl = DataLoader(self.ds, batch_size = self.batch_size, shuffle=True, pin_memory=True)

    def train(self, num_epoch):
        self.model.train()
        
        for _ in range(num_epoch):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(self.dl):
                self.opt.zero_grad()
                data, target = data.cuda(), target.cuda()

                output = self.model(data)
                loss = F.mse_loss(output, target)
                total_loss += loss.item()
                loss.backward()
                self.opt.step()
                bias = self.current_task_idx != 0 # don't zero out bias for first task
                self.restore_fixed_weights(bias=bias) # restore fixed weights

            total_loss /= len(self.dl.dataset)
            self.writer.add_scalar('Loss/train', total_loss, self.epoch)

            if self.epoch != 0 and self.epoch % self.ckpt_every == 0:
                milestone = self.epoch // self.ckpt_every
                self.save(milestone)

            self.epoch += 1
