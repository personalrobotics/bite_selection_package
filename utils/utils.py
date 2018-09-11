'''misc utils'''

from __future__ import print_function
from __future__ import division

import math
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.split(os.getcwd())[0])
from config import config


def load_label_map(label_map_filename):
    with open(label_map_filename, 'r') as f:
        content = f.read().splitlines()
        f.close()

    assert content is not None, 'cannot find label map'

    temp = list()
    for line in content:
        line = line.strip()
        if (len(line) > 2 and
                (line.startswith('id') or
                 line.startswith('name'))):
            temp.append(line.split(':')[1].strip())

    label_dict = dict()
    for idx in range(0, len(temp), 2):
        item_id = int(temp[idx])
        item_name = temp[idx + 1][1:-1]
        label_dict[item_id] = item_name

    return label_dict


def get_accuracy(bmask_preds, bmask_targets, rmask_preds, rmask_targets):
    bmask_precision = 0
    bmask_recall = 0

    bmask_preds = bmask_preds.data.cpu().numpy().flatten()
    bmask_targets = bmask_targets.data.cpu().numpy().flatten()

    bmask_preds_pos = bmask_preds < 0.001
    bmask_targets_pos = bmask_targets == 1

    bmask_true_positives = np.sum(bmask_preds_pos == bmask_targets_pos)
    bmask_false_positives = np.sum(~bmask_preds_pos == bmask_targets_pos)
    bmask_false_negatives = np.sum(bmask_preds_pos == ~bmask_targets_pos)

    bmask_precision = bmask_true_positives / (bmask_true_positives + bmask_false_positives)
    bmask_recall = bmask_true_positives / (bmask_true_positives + bmask_false_negatives)

    if config.use_rotation:
        positives = bmask_preds_pos == bmask_targets_pos
        mask = positives.unsqueeze(2).expand_as(rmask_preds)
        rmask_preds = rmask_preds[mask].view(-1, config.angle_res + 1)
        rmask_targets = rmask_targets[positives]

        rmask_preds = rmask_preds.data.cpu().numpy()
        rmask_targets = rmask_targets.data.cpu().numpy()

        rmask_preds = np.argmax(rmask_preds, axis=1)

        rmask_accuracy = np.mean(np.sin(abs(rmask_preds - rmask_targets)))

    return bmask_precision, bmask_recall, rmask_accuracy
