''' general configurations'''

import os


gpu_id = '0'

use_identity = False
use_rotation = True
use_rot_alt = False
use_densenet = True

valid_denseblock_sizes = [[3, 6], [6, 12]]
denseblock_sizes = valid_denseblock_sizes[1]

project_dir = os.path.split(os.getcwd())[0]

project_prefix = 'food_spnet_c8'
if use_identity:
    project_prefix += '_identity'
if use_densenet:
    project_prefix += '_dense_{}_{}'.format(*denseblock_sizes)
if not use_rotation:
    project_prefix += '_loc_only'
if use_rotation and use_rot_alt:
    project_prefix += '_rot_alt'

num_classes = 8

mask_size = 17  # grid_shape: (17, 17)
angle_res = 18

# project_prefix += 'a_{}'.format(angle_res)

cropped_img_res = mask_size * 8  # 136

train_batch_size = 32
test_batch_size = 4

dataset_dir = os.path.join(project_dir, 'data')

label_map_filename = os.path.join(dataset_dir, 'food_c8_label_map.pbtxt')
img_dir = os.path.join(dataset_dir, 'bounding_boxes_c8/images')
cropped_img_dir = os.path.join(dataset_dir, 'skewering_positions_c8/cropped_images')
mask_dir = os.path.join(dataset_dir, 'skewering_positions_c8/masks')

train_list_filename = os.path.join(dataset_dir, 'food_c8_ann_train.txt')
test_list_filename = os.path.join(dataset_dir, 'food_c8_ann_test.txt')

pretrained_dir = os.path.join(project_dir, 'pretrained')
pretrained_filename = os.path.join(
    pretrained_dir, '{}_net.pth'.format(project_prefix))

checkpoint_filename = os.path.join(
    project_dir, 'checkpoint/{}_ckpt.pth'.format(project_prefix))

checkpoint_best_filename = os.path.join(
    project_dir, 'checkpoint/{}_ckpt_best.pth'.format(project_prefix))
