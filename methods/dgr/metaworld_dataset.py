import torch
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms as T

# mean = torch.load('/scratch/cluster/william/metaworld/state_data/mean.pt')
# std = torch.load('/scratch/cluster/william/metaworld/state_data/std.pt')
# std = std + 1e-9 # avoid division by zero

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
                #rollout[i][0] = (torch.tensor(rollout[i][0], dtype=torch.float32) - mean) / std # normalize
                rollout[i][0] = torch.tensor(rollout[i][0], dtype=torch.float32) # don't normalize
            self.demos += rollout

    def __len__(self):
        return len(self.demos) 

    def __getitem__(self, idx):
        sample = self.demos[idx]
        return sample

    def add_item(self, item):
        self.demos.append(item)

class ImageDataset(Dataset):
    def __init__(self, folder):
        super().__init__()
        exts = ['rollout']
        paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.data = []
        for path in paths:
            rollout = torch.load(path) 
            # only keep the observations and throw away the actions
            for i in range(len(rollout)):
                rollout[i] = torch.tensor(rollout[i][0], dtype=torch.float32)
            self.data += rollout

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def add_item(self, item):
        assert item.shape == (49,)
        self.data.append(item)

def get_videos(rollout, vid_len):
    """parse a rollout into videos of length vid_len"""
    videos = []
    for j in range(len(rollout) - vid_len + 1):
        demo = rollout[j : j + vid_len]
        demo = torch.stack(demo)
        videos.append(demo)

    return videos
    
class VideoDataset(Dataset):
    def __init__(self, folder, num_frames = 10):
        super().__init__()
        exts = ['rollout']
        paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.data = []
        for path in paths:
            rollout = torch.load(path) 
            # only keep the observations and throw away the actions
            for i in range(len(rollout)):
                #rollout[i] = (torch.tensor(rollout[i][0], dtype=torch.float32) - mean) / std
                rollout[i] = torch.tensor(rollout[i][0], dtype=torch.float32)
            self.data += get_videos(rollout, num_frames)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def add_item(self, item):
        self.data.append(item)