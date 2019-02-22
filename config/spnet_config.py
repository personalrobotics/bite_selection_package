''' general configurations'''

import os


use_cuda = True
gpu_id = '0'

use_rotation = True
use_densenet = True

block_config = [3, 6]

project_dir = os.path.split(os.getcwd())[0]

num_classes = 6
mask_size = 17  # grid_shape: (17, 17)
angle_res = presets[pidx]['angle_res']

project_prefix = 'food_spnet_c{}'.format(num_classes)
if use_identity:
    project_prefix += '_identity'
if use_densenet:
    if denseblock_version == 2:
        project_prefix += '_dense_v2'
    else:
        project_prefix += '_dense_{}_{}'.format(*block_config)
if not use_rotation:
    project_prefix += '_loc_only'
else:
    if use_rot_alt:
        project_prefix += '_rot_alt'
    else:
        project_prefix += '_a_{}'.format(angle_res)

###############################################################################
# pw tests: 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.20, 0.30, 0.40
p_weight = 0.08
###############################################################################

cropped_img_res = mask_size * 8  # 136

train_batch_size = 32
test_batch_size = 4

dataset_dir = os.path.join(project_dir, 'data')

label_map_filename = os.path.join(
    dataset_dir, 'food_c{}_label_map.pbtxt'.format(num_classes))
img_dir = os.path.join(
    dataset_dir, 'bounding_boxes_c{}/images'.format(num_classes))
cropped_img_dir = os.path.join(
    dataset_dir, 'skewering_positions_c{}/cropped_images'.format(num_classes))
mask_dir = os.path.join(
    dataset_dir, 'skewering_positions_c{}/masks'.format(num_classes))

train_list_filename = os.path.join(
    dataset_dir, 'food_c{}_ann_train.txt'.format(num_classes))
test_list_filename = os.path.join(
    dataset_dir, 'food_c{}_ann_test.txt'.format(num_classes))

pretrained_dir = os.path.join(project_dir, 'pretrained')
pretrained_filename = os.path.join(
    pretrained_dir, '{}_net.pth'.format(project_prefix))

checkpoint_filename = os.path.join(
    project_dir, 'checkpoint/{}_ckpt.pth'.format(project_prefix))

checkpoint_best_filename = os.path.join(
    project_dir, 'checkpoint/{}_ckpt_best.pth'.format(project_prefix))
