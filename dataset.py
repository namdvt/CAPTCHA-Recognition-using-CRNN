import random

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import glob
import os
import matplotlib.pyplot as plt


def split_image(image):
    output = torch.Tensor([])
    for i in range(0, 200, 10):
        output = torch.cat([output, image[0][:, 0:10].unsqueeze(0)], dim=0)

    return output


class CaptchaImagesDataset(Dataset):
    def __init__(self, root, augment=False):
        super(CaptchaImagesDataset, self).__init__()
        self.root = root
        self.augment = augment

        self.image_list = []
        for ext in ('*.png', '*.jpg'):
            self.image_list.extend(glob.glob(os.path.join(root, ext)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = self.image_list[index]
        text = image.split('/')[-1].split('.')[0]

        image = Image.open(image).convert('RGB')
        image = F.to_tensor(image)

        return image, text


def get_loader(root, batch_size):
    train_dataset = CaptchaImagesDataset(root + '/train', augment=True)
    val_dataset = CaptchaImagesDataset(root + '/val', augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, val_loader


if __name__ == '__main__':
    train, val = get_loader('data/CAPTCHA Images/', batch_size=2)
    for image, labels in train:
        print()
    print()
