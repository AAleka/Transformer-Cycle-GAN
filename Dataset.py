import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ABDataset(Dataset):
    def __init__(self, root_a, root_b=None, transform=None):
        self.root_a = root_a
        self.root_b = root_b
        self.transform = transform

        self.a_images = os.listdir(root_a)
        self.b_images = os.listdir(root_b)
        self.length_dataset = max(len(self.a_images), len(self.b_images))
        self.a_len = len(self.a_images)
        self.b_len = len(self.b_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        if self.root_b is not None:
            a_img = self.a_images[index % self.a_len]
            a_path = os.path.join(self.root_a, a_img)
            a_img = np.array(Image.open(a_path).convert("RGB"))

            b_img = self.b_images[index % self.b_len]
            b_path = os.path.join(self.root_b, b_img)
            b_img = np.array(Image.open(b_path).convert("RGB"))

            if self.transform:
                augmentations = self.transform(image0=a_img, image=b_img)
                a_img = augmentations["image0"]
                b_img = augmentations["image"]

            return a_img, b_img

        elif self.root_b is None:
            a_img = self.a_images[index % self.a_len]
            a_path = os.path.join(self.root_a, a_img)
            a_img = np.array(Image.open(a_path).convert("RGB"))

            if self.transform:
                augmentations = self.transform(image=a_img)
                a_img = augmentations["image"]

            return a_img
