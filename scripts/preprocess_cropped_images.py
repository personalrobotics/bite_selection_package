from __future__ import division

import os
import numpy as np

from PIL import Image


def preprocess_skewering_positions_datset():
    print('preprocess_skewering_positions_datset')

    org_base_dir = '../data/skewering_positions'
    ann_base_dir = os.path.join(org_base_dir, 'annotations')
    img_base_dir = os.path.join(org_base_dir, 'cropped_images')

    processed_base_dir = '../data/processed'
    processed_ann_base_dir = os.path.join(processed_base_dir, 'annotations')
    processed_img_base_dir = os.path.join(processed_base_dir, 'cropped_images')

    if not os.path.exists(processed_ann_base_dir):
        os.makedirs(processed_ann_base_dir)
    if not os.path.exists(processed_img_base_dir):
        os.makedirs(processed_img_base_dir)

    anns = os.listdir(ann_base_dir)

    target_size = 56

    summary_list = list()

    for idx in range(len(anns)):
        ann_filename = anns[idx]
        if not ann_filename.endswith('.out'):
            continue

        img_filename = '{}.jpg'.format(ann_filename[:-4])

        if (not os.path.exists(os.path.join(img_base_dir, img_filename)) or
            not os.path.exists(os.path.join(ann_base_dir, ann_filename))):
            continue

        img = Image.open(os.path.join(img_base_dir, img_filename))

        ratio = float(target_size / max(img.size))
        new_size = tuple([int(x * ratio) for x in img.size])
        pads = [(target_size - new_size[0]) // 2,
                (target_size - new_size[1]) // 2]

        img = img.resize(new_size, Image.ANTIALIAS)

        new_img = Image.new('RGB', (target_size, target_size))
        new_img.paste(img, pads)

        with open(os.path.join(ann_base_dir, ann_filename), 'r') as f:
            ann = f.read().split()
            f.close()

        for i in range(2):
            ann[i] = np.round(float(ann[i]) * ratio + pads[i])

        new_ann_str = '{0:.2f} {1:.2f} {2}'.format(*ann)
        with open(os.path.join(processed_ann_base_dir, ann_filename), 'w') as f:
            f.write(new_ann_str)
            f.close()

        new_img.save(os.path.join(processed_img_base_dir, img_filename), 'JPEG')
        print(img_filename)

        summary_list.append(
            '{} {}'.format(img_filename, new_ann_str))

    max_train_idx = int(len(summary_list) * 0.9)

    with open(os.path.join(processed_base_dir, 'sp_train.txt'), 'w') as f:
        for idx in range(max_train_idx):
            f.write('{}\n'.format(summary_list[idx]))
        f.close()

    with open(os.path.join(processed_base_dir, 'sp_test.txt'), 'w') as f:
        for idx in range(max_train_idx, len(summary_list)):
            f.write('{}\n'.format(summary_list[idx]))
        f.close()

    print('finished\n')


if __name__ == '__main__':
    preprocess_skewering_positions_datset()
