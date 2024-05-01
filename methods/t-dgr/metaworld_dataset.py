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
                rollout[i] = [torch.tensor(rollout[i][0], dtype=torch.float32), torch.tensor([i])]
            self.data += rollout

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def add_item(self, item):
        assert len(item) == 2 and item[0].shape == (49,) and item[1].shape == (1,)
        self.data.append(item)

def get_videos(rollout, vid_len):
    """parse a rollout into videos of length vid_len"""
    videos = []
    for j in range(len(rollout) - vid_len + 1):
        demo = rollout[j : j + vid_len]
        demo = torch.stack(demo)
        assert demo.shape[0] == vid_len and demo.shape[1] == 49
        videos.append([demo, torch.full((1,), j)])

    return videos
    
class VideoDataset(Dataset):
    def __init__(self, folder, num_frames = 16):
        super().__init__()
        exts = ['rollout']
        paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.data = []
        for path in paths:
            rollout = torch.load(path) 
            # only keep the observations and throw away the actions
            for i in range(len(rollout)):
                rollout[i] = torch.tensor(rollout[i][0], dtype=torch.float32)
            self.data += get_videos(rollout, num_frames)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def add_item(self, item):
        assert len(item) == 2 and item[0].shape == (16, 49) and item[1].shape == (1,)
        self.data.append(item)