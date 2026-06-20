import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class LandDataset(Dataset):

    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))

        self.mapping = {
            89: 0,
            104: 1,
            106: 2,
            169: 3,
            184: 4
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_path = os.path.join(
            self.image_dir,
            self.images[index]
        )

        mask_path = os.path.join(
            self.mask_dir,
            self.images[index]
        )

        image = cv2.imread(img_path)
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = cv2.resize(
            image,
            (256,256)
        )

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        mask = cv2.resize(
            mask,
            (256,256),
            interpolation=cv2.INTER_NEAREST
        )

        new_mask = np.zeros_like(mask)

        for old_val, new_val in self.mapping.items():
            new_mask[mask == old_val] = new_val

        image = torch.tensor(
            image,
            dtype=torch.float32
        ).permute(2,0,1)/255.0

        mask = torch.tensor(
            new_mask,
            dtype=torch.long
        )

        return image, mask