''' general configurations'''

import os


use_cuda = True
gpu_id = '1'

use_rgb = True
use_depth = not use_rgb
assert use_rgb or use_depth, 'invalid configuration'

use_densenet = False
denseblock_sizes = [3, 6]

project_dir = os.path.split(os.getcwd())[0]

# grapes, cherry_tomatoes, broccoli, cauliflower, honeydew,
# banana, kiwi, strawberry, cantaloupe, carrots, celeries,
# apples, bell_pepper
excluded_item = 'broccoli'

project_prefix = 'food_spactionnet_{}{}{}{}'.format(
    'rgb' if use_rgb else '',
    'd' if use_depth else '',
    '_dense' if use_densenet else '',
    '_wo_{}'.format(excluded_item) if excluded_item else '')

img_res = 9 * 16  # 144

# [p1_x, p1_y, p2_x, p2_y, a1, a2, a3, a4, a5, a6]
final_vector_size = 10

train_batch_size = 8
test_batch_size = 4

dataset_dir = os.path.join(project_dir, 'data')
sub_dir = 'skewering_positions_general'

label_map_filename = os.path.join(
    dataset_dir, 'food_general_label_map.pbtxt')
img_dir = os.path.join(
    dataset_dir, sub_dir, 'cropped_images')
depth_dir = os.path.join(
    dataset_dir, sub_dir, 'cropped_depth')
ann_dir = os.path.join(
    dataset_dir, sub_dir, 'annotations')
success_rate_map_path = os.path.join(
    dataset_dir, 'identity_to_success_rate_map.json')

train_list_filepath = os.path.join(
    dataset_dir, sub_dir, 'train.txt')
test_list_filepath = os.path.join(
    dataset_dir, sub_dir, 'test.txt')

pretrained_dir = os.path.join(project_dir, 'pretrained')
pretrained_filename = os.path.join(
    pretrained_dir, '{}_net.pth'.format(project_prefix))

checkpoint_filename = os.path.join(
    project_dir, 'checkpoint/{}_ckpt.pth'.format(project_prefix))

checkpoint_best_filename = os.path.join(
    project_dir, 'checkpoint/{}_ckpt_best.pth'.format(project_prefix))
