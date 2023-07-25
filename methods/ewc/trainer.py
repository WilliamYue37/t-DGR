import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import os

class Trainer():
    def __init__(self, model, dataset, train_batch_size = 32, train_lr = 1e-4, ewc_lambda = 400, ckpt_every = 10, ckpts_folder = './ckpts'):
        self.model = model

        self.writer = SummaryWriter(log_dir=ckpts_folder + '/learner_logs')
        self.ckpt_every = ckpt_every
        self.ckpts_folder = Path(ckpts_folder + '/learner_ckpts')

        self.batch_size = train_batch_size

        self.ds = dataset
        self.dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True)

        self.opt = Adam(model.parameters(), lr = train_lr)

        self.epoch = 0
        self.is_first_task = True

        # ewc stuff
        self.ewc_lambda = ewc_lambda
        self.fisher_dict = {} # fisher approximation
        self.optpar_dict = {} # optimal parameters

    def save(self, milestone):
        "save model: string is for special milestones, int is for regular milestones"
        data = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'fisher': self.fisher_dict,
            'optpar': self.optpar_dict,
            'first_task': self.is_first_task,
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
        self.fisher_dict = data['fisher']
        self.optpar_dict = data['optpar']
        self.is_first_task = data['first_task']

    def load_new_dataset(self, dataset):
        self.ds = dataset
        self.dl = DataLoader(self.ds, batch_size = self.batch_size, shuffle=True, pin_memory=True)

    def update_fisher(self):
        for name, param in self.model.named_parameters():
            self.fisher_dict[name] = 0
        
        for batch_idx, (data, target) in enumerate(self.dl):
            self.opt.zero_grad()
            data, target = data.cuda(), target.cuda()

            output = self.model(data)
            loss = F.mse_loss(output, target)

            loss.backward()
            
            for name, param in self.model.named_parameters():
                self.fisher_dict[name] += param.grad.data.clone().pow(2) / len(self.ds)

        for name, param in self.model.named_parameters():
            self.optpar_dict[name] = param.data.clone()

    def train(self, num_epoch):
        self.model.train()
        for _ in range(num_epoch):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(self.dl):
                self.opt.zero_grad()
                data, target = data.cuda(), target.cuda()

                output = self.model(data)
                loss = F.mse_loss(output, target)

                # add ewc loss
                if not self.is_first_task:
                    for name, param in self.model.named_parameters():
                        fisher = self.fisher_dict[name]
                        optpar = self.optpar_dict[name]
                        loss += (fisher * (optpar - param).pow(2)).sum() * self.ewc_lambda

                total_loss += loss.item()
                loss.backward()
                self.opt.step()

            total_loss /= len(self.dl.dataset)
            self.writer.add_scalar('Loss/train', total_loss, self.epoch)

            if self.epoch != 0 and self.epoch % self.ckpt_every == 0:
                milestone = self.epoch // self.ckpt_every
                self.save(milestone)

            self.epoch += 1
        
        self.update_fisher()
        self.is_first_task = False

            

