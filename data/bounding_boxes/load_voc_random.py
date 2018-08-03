#!/usr/bin/env python3

import os
import shutil


def load_voc_random():
    print('load_voc_random')

    voc_base_dir = os.path.expanduser('~/external/Data/VOCdevkit/VOC2012')
    target_base_dir = os.path.join(voc_base_dir, 'ImageSets/Main/')
    target_txt = os.path.join(target_base_dir, 'diningtable_trainval.txt')

    image_base_dir = os.path.join(voc_base_dir, 'JPEGImages')

    save_dir = './voc_random'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(target_txt, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.split(' ')
            name = items[0].strip()
            val = int(items[-1].strip())
            if val == 1:
                print(name, val)
                image_filename = os.path.join(image_base_dir, name) + '.jpg'
                shutil.copy(
                    image_filename,
                    os.path.join(save_dir, '{}.jpg'.format(name)))
        f.close()


if __name__ == '__main__':
    load_voc_random()
