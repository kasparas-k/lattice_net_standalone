from pathlib import Path

import numpy as np
import torch
from torch.utils import data

class SemPOSS(data.Dataset):
    def __init__(self, data_path, mode='train', maxpoints=None):
        self.mode = mode
        self.maxpoints = maxpoints
        xyz = []
        labels = []
        for seq in [f for f in (Path(data_path) / self.mode).iterdir() if f.is_dir()]:
            xyz += sorted(list((seq / 'velodyne').glob('*.bin')))
            labels += sorted(list((seq / 'labels').glob('*.label')))

        self.xyz = xyz
        self.labels = labels

    def __len__(self):
        return len(self.xyz)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.xyz[index], dtype=np.float32).reshape((-1, 4))
        xyz = raw_data[:,:3]

        labels = np.fromfile(self.labels[index], dtype=np.int32)
        labels = labels & 0xFFFF  # delete high 16 bits
        if self.maxpoints is not None:
            if self.maxpoints > 0 and len(labels) > self.maxpoints:
                idx = list(range(len(labels)))
                chosen = np.random.choice(idx, self.maxpoints, replace=False)
                xyz = xyz[chosen]
                labels = labels[chosen]

        return xyz, labels
