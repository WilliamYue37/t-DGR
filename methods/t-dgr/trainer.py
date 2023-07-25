import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import os

class Trainer():
    def __init__(self, model, dataset, train_batch_size = 64, train_lr = 1e-4, ckpt_every = 500, ckpts_folder = './ckpts'):
        self.model = model

        self.writer = SummaryWriter(log_dir=ckpts_folder + '/learner_logs')
        self.ckpt_every = ckpt_every
        self.ckpts_folder = Path(ckpts_folder + '/learner_ckpts')

        self.batch_size = train_batch_size

        self.ds = dataset
        self.dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True)

        self.opt = Adam(model.parameters(), lr = train_lr)

        self.epoch = 0

    def save(self, milestone):
        "save model: string is for special milestones, int is for regular milestones"
        data = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
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

            total_loss /= len(self.dl.dataset)
            self.writer.add_scalar('Loss/train', total_loss, self.epoch)

            if self.epoch != 0 and self.epoch % self.ckpt_every == 0:
                milestone = self.epoch // self.ckpt_every
                self.save(milestone)

            self.epoch += 1
