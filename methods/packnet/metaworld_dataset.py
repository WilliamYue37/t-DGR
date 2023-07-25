import torch
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms as T

class MetaworldDataset(Dataset):
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
