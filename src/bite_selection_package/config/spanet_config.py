''' general configurations'''

import os


use_cuda = True
gpu_id = '0'

use_rgb = True
use_depth = False  # not use_rgb
assert use_rgb or use_depth, 'invalid configuration'

use_densenet = False

# Pretrained block configs:
# densenet121 (6, 12, 24, 16)
# densenet169 (6, 12, 32, 32)
# densenet201 (6, 12, 48, 32)
# densenet161 (6, 12, 36, 24)
block_config = [3, 6, 12]

project_dir = os.path.split(os.getcwd())[0]

items = [None,
         'apple', 'banana', 'bell_pepper', 'broccoli', 'cantaloupe',
         'carrot', 'cauliflower', 'celery', 'cherry_tomato', 'grape',
         'honeydew', 'kiwi', 'melon', 'strawberry']
excluded_item = items[0]


img_res = 9 * 16  # 144

# [p1_x, p1_y, p2_x, p2_y, a1, a2, a3, a4, a5, a6]
final_vector_size = 10

train_batch_size = 33
test_batch_size = 4

###############################################################################

project_keyword = 'spanet'

project_prefix = 'food_{}_{}{}{}{}'.format(
    project_keyword,
    'rgb' if use_rgb else '',
    'd' if use_depth else '',
    '_dense' if use_densenet else '',
    '_wo_{}'.format(excluded_item) if excluded_item else '')

dataset_dir = os.path.join(
    project_dir, 'data/skewering_positions_{}'.format(project_keyword))

img_dir = os.path.join(dataset_dir, 'cropped_images')
depth_dir = os.path.join(dataset_dir, 'cropped_depth')
ann_dir = os.path.join(dataset_dir, 'annotations')
success_rate_map_path = os.path.join(
    dataset_dir,
    'identity_to_success_rate_map_{}.json'.format(project_keyword))

train_list_filepath = os.path.join(dataset_dir, 'train.txt')
test_list_filepath = os.path.join(dataset_dir, 'test.txt')

###############################################################################

pretrained_dir = os.path.join(project_dir, 'pretrained')
pretrained_filename = os.path.join(
    pretrained_dir, '{}_net.pth'.format(project_prefix))

checkpoint_filename = os.path.join(
    project_dir, 'checkpoint/{}_ckpt.pth'.format(project_prefix))

checkpoint_best_filename = os.path.join(
    project_dir, 'checkpoint/{}_ckpt_best.pth'.format(project_prefix))

###############################################################################

def set_project_prefix():
    global project_prefix, pretrained_filename
    global checkpoint_filename, checkpoint_best_filename

    project_prefix = 'food_{}_{}{}{}{}'.format(
        project_keyword,
        'rgb' if use_rgb else '',
        'd' if use_depth else '',
        '_dense' if use_densenet else '',
        '_wo_{}'.format(excluded_item) if excluded_item else '')

    pretrained_filename = os.path.join(
        pretrained_dir, '{}_net.pth'.format(project_prefix))

    checkpoint_filename = os.path.join(
        project_dir, 'checkpoint/{}_ckpt.pth'.format(project_prefix))

    checkpoint_best_filename = os.path.join(
        project_dir, 'checkpoint/{}_ckpt_best.pth'.format(project_prefix))
