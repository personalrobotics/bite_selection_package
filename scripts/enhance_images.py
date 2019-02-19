#!/usr/bin/env python3

from __future__ import division

import numpy as np
import os
import glob

from PIL import Image, ImageEnhance
from multiprocessing import Pool


img_dir = '../data/bounding_boxes_general/images'


def process_image(img_path):
    img = Image.open(img_path)
    w, h = img.size
    img_gray = img.crop((170, 90, w - 170, h - 90)).convert('LA')

    pixel_mean = np.mean(img_gray)
    b_ratio = 160.0 / pixel_mean
    b_ratio = b_ratio ** 3 if b_ratio > 1.0 else b_ratio
    c_ratio = 180.0 / pixel_mean

    img = ImageEnhance.Brightness(img).enhance(b_ratio)
    img = ImageEnhance.Contrast(img).enhance(c_ratio)

    img_name = os.path.basename(img_path)
    save_path = os.path.join(img_dir, '{}_mod{}'.format(
        img_name[:-4], img_name[-4:]))
    img.save(save_path)
    print(save_path)


def enhance_images():
    print('enhance_images')
    img_paths = glob.glob(os.path.join(img_dir, '*.png'))

    target_paths = list()
    for img_path in img_paths:
        if img_path.endswith('_mod.png'):
            continue
        target_paths.append(img_path)

    pool = Pool(8)
    pool.map(process_image, target_paths)


if __name__ == '__main__':
    enhance_images()
