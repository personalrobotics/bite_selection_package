from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import os
import numpy as np
import pandas
import random

import torch
import torch.utils.data as data

from PIL import Image, ImageEnhance, ImageFilter

import bite_selection_package.utils.transform as trans
from bite_selection_package.utils.utils import load_label_map
from bite_selection_package.config import spnet_config as config


class SPDataset(data.Dataset):
    def __init__(self,
                 cropped_img_dir=config.cropped_img_dir,
                 mask_dir=config.mask_dir,
                 list_filename=config.train_list_filename,
                 label_map_filename=config.label_map_filename,
                 train=True,
                 exp_mode='nothing',  # 'exclude', 'test', others
                 transform=None,
                 cropped_img_res=config.cropped_img_res):

        self.cropped_img_dir = cropped_img_dir
        self.mask_dir = mask_dir
        self.train = train
        self.transform = transform
        self.cropped_img_res = cropped_img_res

        self.label_map = load_label_map(label_map_filename)

        self.cropped_filenames = list()
        self.labels = list()
        self.bmasks = list()
        self.rmasks = list()

        with open(list_filename) as f:
            lines = f.readlines()
            f.close()

        self.num_samples = 0
        isize = 6
        for line in lines:
            splited = line.strip().split()
            this_img_filename = splited[0]

            num_boxes = (len(splited) - 1) // isize

            for bidx in range(num_boxes):
                xmin = int(splited[1 + isize * bidx])
                ymin = int(splited[2 + isize * bidx])
                cls = int(splited[5 + isize * bidx])
                cidx = splited[6 + isize * bidx]

                food_identity = self.label_map[cls]
                if exp_mode == 'exclude':
                    if food_identity == config.excluded_item:
                        continue
                elif exp_mode == 'test':
                    if food_identity != config.excluded_item:
                        continue

                cropped_filename = os.path.join(
                    self.cropped_img_dir, '{0}_{1}_{2:04d}{3:04d}.jpg'.format(
                        this_img_filename[:-4], self.label_map[cls],
                        xmin, ymin))

                mask_filename = os.path.join(
                    self.mask_dir, '{0}_{1}_{2:04d}{3:04d}.txt'.format(
                        this_img_filename[:-4], self.label_map[cls],
                        xmin, ymin))

                if not os.path.exists(mask_filename):
                    continue

                this_mask = np.asarray(
                    pandas.read_csv(mask_filename, header=None).values)

                # binary mask
                this_bmask = np.zeros_like(this_mask)
                this_bmask[this_mask > -1] = 1

                if np.sum(this_bmask) == 0:
                    continue

                # rotation mask
                this_rmask = this_mask.copy()
                this_rmask[this_rmask == 0] = -1
                this_rmask[this_rmask > 0] /= 180 / config.angle_res
                this_rmask = np.round(this_rmask).astype(np.long) + 1
                this_rmask[this_rmask > config.angle_res] = 1

                self.cropped_filenames.append(cropped_filename)
                self.labels.append(cls)
                self.bmasks.append(this_bmask.flatten())
                self.rmasks.append(this_rmask.flatten())

                self.num_samples += 1

    def __getitem__(self, idx):
        cropped_filename = self.cropped_filenames[idx]
        img_org = Image.open(os.path.join(
            self.cropped_img_dir, cropped_filename))
        if img_org.mode != 'RGB':
            img_org = img_org.convert('RGB')

        label = self.labels[idx]
        bmask = self.bmasks[idx].copy()
        rmask = self.rmasks[idx].copy()

        target_size = self.cropped_img_res
        ratio = float(target_size / max(img_org.size))
        new_size = tuple([int(x * ratio) for x in img_org.size])
        pads = [(target_size - new_size[0]) // 2,
                (target_size - new_size[1]) // 2]
        img_org = img_org.resize(new_size, Image.ANTIALIAS)

        img = Image.new('RGB', (target_size, target_size))
        img.paste(img_org, pads)

        gt_bmask = torch.Tensor(bmask)
        gt_rmask = torch.Tensor(rmask)

        # Data augmentation
        if self.train:
            img, gt_bmask, gt_rmask = trans.random_flip_w_mask(
                img, gt_bmask, gt_rmask, config.angle_res)
            if random.random() > 0.5:
                img = ImageEnhance.Color(img).enhance(
                    random.uniform(0, 1))
                img = ImageEnhance.Brightness(img).enhance(
                    random.uniform(0.4, 2))
                img = ImageEnhance.Contrast(img).enhance(
                    random.uniform(0.4, 1.5))
                img = ImageEnhance.Sharpness(img).enhance(
                    random.uniform(0.4, 1.5))

        img = self.transform(img)

        return img, gt_bmask, gt_rmask

    def collate_fn(self, batch):
        imgs = [x[0] for x in batch]
        bmasks = [x[1] for x in batch]
        rmasks = [x[2] for x in batch]

        return (torch.stack(imgs),
                torch.stack(bmasks),
                torch.stack(rmasks))

    def __len__(self):
        return self.num_samples


def test():
    print('[spdataset] test')
    ds = SPDataset(
        list_filename=config.test_list_filename,
        train=False)
    import IPython; IPython.embed()


if __name__ == '__main__':
    test()
