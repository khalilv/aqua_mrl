from typing import Any
from torch.utils.data import Dataset
import os
import glob
import torch
import numpy as np

class ExpertDataset(Dataset):
    def __init__(self, root, history) -> None:
        self.root = root
        self.history = history
        self.states = sorted(
            glob.glob(os.path.join(self.root, 'states', '*'))
        )
        self.depths = sorted(
            glob.glob(os.path.join(self.root, 'depths', '*'))
        )
        self.actions = sorted(
            glob.glob(os.path.join(self.root, 'actions', '*'))
        )
        return

    def __len__(self) -> int:
        return len(self.states)
    
    def __getitem__(self, idx: int) -> Any:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        state_path = self.states[idx]
        depth_path = self.depths[idx]
        action_path = self.actions[idx]
        with open(state_path, 'rb') as s:
            state = np.load(s)
            recorded_hist = state.shape[0]
            state = state[recorded_hist-self.history:,:,:]
            state = torch.tensor(state, dtype=torch.float32)
        with open(depth_path, 'rb') as d:
            depth = np.load(d)
            recorded_hist = len(depth)
            depth = depth[recorded_hist-self.history:]
            depth = torch.tensor(depth, dtype=torch.float32)
        with open(action_path, 'rb') as a:
            action = np.load(a)
            action = torch.tensor(action, dtype=torch.float32)

        sample = {"state": state, "depth": depth, "action": action}
        
        return sample    