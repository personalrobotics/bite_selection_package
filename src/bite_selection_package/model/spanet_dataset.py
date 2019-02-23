from __future__ import division
from __future__ import print_function

import sys
import os
import json
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image, ImageEnhance, ImageFilter

# import bite_selection_package.utils.transform as trans
from bite_selection_package.config import spanet_config as config


class SPANetDataset(data.Dataset):
    def __init__(self,
                 img_dir=config.img_dir,
                 depth_dir=config.depth_dir,
                 ann_dir=config.ann_dir,
                 list_filepath=None,
                 ann_filenames=None,
                 success_rate_map_path=config.success_rate_map_path,
                 train=True,
                 exp_mode='exclude',  # 'exclude', 'test', others
                 transform=None,
                 img_res=config.img_res):
        if ann_filenames is None:
            assert list_filepath, 'invalid list_filepath'
            with open(list_filepath, 'r') as f_list:
                ann_filenames = list(map(str.strip, f_list.readlines()))
        assert ann_filenames and len(ann_filenames) > 0, 'invalid annotations'

        self.lfp = list_filepath
        self.afn = ann_filenames

        self.img_dir = img_dir
        self.depth_dir = depth_dir
        self.ann_dir = ann_dir
        self.train = train
        self.transform = transform
        self.img_res = img_res

        self.img_filepaths = list()
        self.depth_filepaths = list()
        self.success_rates = list()
        self.end_points = list()
        self.food_identities = list()

        self.num_samples = 0

        with open(success_rate_map_path, 'r') as f_srm:
            srm_str = f_srm.read().strip()
            f_srm.close()
        assert srm_str, 'cannot load success rate map'
        map_configs = json.loads(srm_str)
        self.action_keys = map_configs['action_keys']
        self.success_rate_map = map_configs['success_rates']

        for ann_filename in ann_filenames:
            sidx = 1 if ann_filename.startswith('sample') else 2
            food_identity = '_'.join(
                ann_filename.split('.')[0].split('+')[-1].split('_')[sidx:-1])
            if exp_mode == 'exclude':
                if food_identity == config.excluded_item:
                    continue
            elif exp_mode == 'test':
                if food_identity != config.excluded_item:
                    continue

            ann_filepath = os.path.join(self.ann_dir, ann_filename)

            img_filename = ann_filename[:-4] + '.png'
            img_filepath = os.path.join(self.img_dir, img_filename)
            if not os.path.exists(img_filepath):
                continue

            depth_filepath = os.path.join(self.depth_dir, img_filename)
            if config.use_depth and not os.path.exists(depth_filepath):
                continue

            with open(ann_filepath, 'r') as f_ann:
                values = list(map(float, f_ann.read().strip().split()))
                f_ann.close()
            if values is None or len(values) != 4:
                continue

            p1 = values[:2]
            p2 = values[2:]
            if p1[0] > p2[0]:
                p1, p2 = p2, p1

            self.img_filepaths.append(img_filepath)
            self.depth_filepaths.append(depth_filepath)
            self.success_rates.append(self.success_rate_map[food_identity])
            self.end_points.append((p1, p2))
            self.food_identities.append(food_identity)
            self.num_samples += 1

    def resize_img(self, img_org, image_mode='RGB'):
        target_size = self.img_res
        ratio = float(target_size / max(img_org.size))
        new_size = tuple([int(x * ratio) for x in img_org.size])
        pads = [(target_size - new_size[0]) // 2,
                (target_size - new_size[1]) // 2]
        img_org = img_org.resize(new_size, Image.ANTIALIAS)

        img = Image.new(image_mode, (target_size, target_size))
        img.paste(img_org, pads)
        return img

    def __getitem__(self, idx):
        depth_img, rgb_img = None, None
        if config.use_depth:
            depth_filepath = self.depth_filepaths[idx]
            depth_img = Image.open(depth_filepath)
            if depth_img.mode != 'F':
                depth_img = depth_img.convert('F')
            depth_img = self.resize_img(depth_img, 'F')
        if config.use_rgb:
            rgb_filepath = self.img_filepaths[idx]
            rgb_img = Image.open(rgb_filepath)
            if rgb_img.mode != 'RGB':
                rgb_img = rgb_img.convert('RGB')
            rgb_img = self.resize_img(rgb_img, 'RGB')

        this_end_points = self.end_points[idx]
        this_success_rates = self.success_rates[idx]

        gt_vector = list()
        gt_vector.extend(this_end_points[0])
        gt_vector.extend(this_end_points[1])
        gt_vector.extend(this_success_rates)
        gt_vector = torch.Tensor(gt_vector)

        # Data augmentation
        if self.train:
            if config.use_rgb and random.random() > 0.5:
                rgb_img = ImageEnhance.Color(rgb_img).enhance(
                    random.uniform(0, 1))
                rgb_img = ImageEnhance.Brightness(rgb_img).enhance(
                    random.uniform(0.4, 2.5))
                rgb_img = ImageEnhance.Contrast(rgb_img).enhance(
                    random.uniform(0.4, 2.0))
                rgb_img = ImageEnhance.Sharpness(rgb_img).enhance(
                    random.uniform(0.4, 1.5))

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()])
        if config.use_rgb:
            rgb_img = self.transform(rgb_img)
        if config.use_depth:
            depth_img = self.transform(depth_img)

        return rgb_img, depth_img, gt_vector

    def collate_fn(self, batch):
        rgb_imgs = [x[0] for x in batch]
        depth_imgs = [x[1] for x in batch]
        vectors = [x[2] for x in batch]

        rgb_imgs = torch.stack(rgb_imgs) if rgb_imgs[0] is not None else None
        depth_imgs = torch.stack(depth_imgs) if depth_imgs[0] is not None else None
        vectors = torch.stack(vectors)

        return (rgb_imgs, depth_imgs, vectors)

    def __len__(self):
        return self.num_samples


def test():
    print('[spanet_dataset] test')
    ds = SPANetDataset(
        list_filepath=config.train_list_filepath,
        train=True)

    import IPython; IPython.embed()


if __name__ == '__main__':
    test()