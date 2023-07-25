import torch
from torch.utils.data import Dataset
from pathlib import Path

class PolicyDataset(Dataset):
    """Metaworld Demos"""

    def __init__(self, folder):
        super().__init__()
        exts = ['rollout']
        paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.demos = []
        for path in paths:
            rollout = torch.load(path)
            for i in range(len(rollout)):
                rollout[i][0] = torch.tensor(rollout[i][0], dtype=torch.float32)
            self.demos += rollout

    def __len__(self):
        return len(self.demos) 

    def __getitem__(self, idx):
        sample = self.demos[idx]
        return sample

    def add_item(self, item):
        self.demos.append(item)

class DynamicsDataset(Dataset):
    """State-Action-State transitions"""

    def __init__(self, folder):
        super().__init__()
        exts = ['rollout']
        paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.demos = []
        for path in paths:
            rollout = torch.load(path)
            for i in range(len(rollout)):
                rollout[i][0] = torch.tensor(rollout[i][0], dtype=torch.float32)
                rollout[i][1] = torch.tensor(rollout[i][1], dtype=torch.float32)

            for i in range(len(rollout) - 1):
                self.demos.append([rollout[i][0], rollout[i][1], rollout[i+1][0]])

    def __len__(self):
        return len(self.demos) 

    def __getitem__(self, idx):
        sample = self.demos[idx]
        return sample

    def add_item(self, item):
        self.demos.append(item)
    
class StartStateDataset(Dataset):
    """Start states"""
    def __init__(self, folder):
        super().__init__()
        exts = ['rollout']
        paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.data = []
        for path in paths:
            rollout = torch.load(path) 
            self.data.append(torch.tensor(rollout[0][0], dtype=torch.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def add_item(self, item):
        self.data.append(item)