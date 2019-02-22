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
angle_res = 18

cropped_img_res = mask_size * 8  # 136

# pw tests: 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.20, 0.30, 0.40
p_weight = 0.08

train_batch_size = 32
test_batch_size = 4

###############################################################################

project_keyword = 'spnet_all'  # 'c{}'.format(num_classes)

project_prefix = 'food_{}'.format(project_keyword)
if use_densenet:
    project_prefix += '_dense_{}_{}'.format(*block_config)
if use_rotation:
    project_prefix += '_a_{}'.format(angle_res)
else:
    project_prefix += '_loc_only'

dataset_dir = os.path.join(project_dir, 'data')

label_map_filename = os.path.join(
    dataset_dir, 'bounding_boxes_{}/food_{}_label_map.pbtxt'.format(
        project_keyword, project_keyword))

cropped_img_dir = os.path.join(
    dataset_dir,
    'skewering_positions_{}/cropped_images'.format(project_keyword))
mask_dir = os.path.join(
    dataset_dir,
    'skewering_positions_{}/masks'.format(project_keyword))

train_list_filename = os.path.join(
    dataset_dir, 'bounding_boxes_{}/food_{}_ann_train.txt'.format(
        project_keyword, project_keyword))
test_list_filename = os.path.join(
    dataset_dir, 'bounding_boxes_{}/food_{}_ann_test.txt'.format(
        project_keyword, project_keyword))

###############################################################################

pretrained_dir = os.path.join(project_dir, 'pretrained')
pretrained_filename = os.path.join(
    pretrained_dir, '{}_net.pth'.format(project_prefix))

checkpoint_filename = os.path.join(
    project_dir, 'checkpoint/{}_ckpt.pth'.format(project_prefix))

checkpoint_best_filename = os.path.join(
    project_dir, 'checkpoint/{}_ckpt_best.pth'.format(project_prefix))
