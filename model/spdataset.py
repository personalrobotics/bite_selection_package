from __future__ import print_function

import os

import torch.utils.data as data

from PIL import Image


class SPDataset(data.Dataset):
    def __init__(self, root, list_file, train, transform, input_size):
        self.root = root
        self.train = train
        self.transform = transform
        self.input_size = input_size

        self.img_filenames = list()
        self.gt_positions = list()
        self.gt_angles = list()

        with open(list_file) as f:
            lines = f.readlines()
            self.num_samples = len(lines)

    def __getitem__(self, idx):
        img_filename = self.img_filenames[idx]
        img = Image.open(os.path.join(self.root, img_filename))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        gt_position = self.gt_positions[idx]
        gt_angle = self.gt_angles[idx]
        size = self.input_size

        # TODO: resize and pad img

        img = self.transform(img)
        return img, gt_position, gt_angle

    def collate_fn(self, batch):
        imgs = [x[0] for x in batch]
        positions = [x[1] for x in batch]
        angles = [x[2] for x in batch]
        return imgs, positions, angles

    def __len__(self):
        return self.num_samples
