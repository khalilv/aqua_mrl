from typing import Any
from torch.utils.data import Dataset
import os
import glob
import torch
import cv2  

class SegmentationDataset(Dataset):
    def __init__(self, root, split, transforms) -> None:
        self.root = os.path.join(root, split)
        self.transforms = transforms
        self.images = sorted(
            glob.glob(os.path.join(self.root, 'images', '*'))
        )
        self.masks = sorted(
            glob.glob(os.path.join(self.root, 'masks', '*'))
        )
        return

    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Any:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        im = cv2.imread(image_path)
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.transforms is not None:
            sample = self.transforms(image=im, mask=mask)
        else:
            sample = {"image": im, "mask": mask}
        
        return sample    