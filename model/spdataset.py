from __future__ import division
from __future__ import print_function

import os
import numpy as np

import torch
import torch.utils.data as data

from PIL import Image

import sys
sys.path.append('../')

import utils.transform as trans


class SPDataset(data.Dataset):
    def __init__(self, root, list_file, train, transform, img_size):
        self.root = root
        self.train = train
        self.transform = transform
        self.img_size = img_size

        self.img_filenames = list()
        self.gt_positions = list()
        self.gt_angles = list()

        with open(list_file) as f:
            lines = f.readlines()
            self.num_samples = len(lines)

            for line in lines:
                items = line.split()

                self.img_filenames.append(items[0])

                x = float(items[1]) / img_size
                y = float(items[2]) / img_size
                # pos = int(x * 8) + int(y * 8) * 8

                self.gt_positions.append([x, y])

                ang = np.round(float(items[3]) / 10)
                if ang >= 18:
                    ang = 0
                self.gt_angles.append(ang)

    def __getitem__(self, idx):
        img_filename = self.img_filenames[idx]
        img = Image.open(os.path.join(self.root, img_filename))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        gt_loc = self.gt_positions[idx]
        gt_rot = self.gt_angles[idx]
        size = self.img_size

        # TODO: resize and pad img

        # Data augmentation
        if self.train:
            img, gt_loc, gt_rot = trans.random_flip(img, gt_loc, gt_rot)
            img, gt_loc, gt_rot = trans.random_rotate(img, gt_loc, gt_rot)

        img = self.transform(img)
        gt_loc = torch.Tensor(gt_loc)
        gt_rot = torch.FloatTensor([gt_rot])
        return img, gt_loc, gt_rot

    def collate_fn(self, batch):
        imgs = [x[0] for x in batch]
        positions = [x[1] for x in batch]
        angles = [x[2] for x in batch]

        return torch.stack(imgs), torch.stack(positions), torch.stack(angles)

    def __len__(self):
        return self.num_samples
