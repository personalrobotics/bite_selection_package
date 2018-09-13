'''misc utils'''

from __future__ import print_function
from __future__ import division

import numpy as np
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

    bp = bmask_preds.data.cpu().numpy().flatten()
    bt = bmask_targets.data.cpu().numpy().flatten()

    pred_labels = bp > 0.05
    true_labels = bt == 1

    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

    bmask_precision = 0 if (TP + FP) == 0 else TP / (TP + FP)
    bmask_recall = 0 if (TP + FN) == 0 else TP / (TP + FN)

    if config.use_rotation:
        if config.use_rot_alt:
            positives = np.logical_and(pred_labels == 1, true_labels == 1)

            rp = rmask_preds.data.cpu().numpy().flatten()
            rt = rmask_targets.data.cpu().numpy().flatten()

            rp_tp = rp[positives]
            rt_tp = rt[positives]

            if len(rp_tp) == 0:
                rmask_dist = -1
            else:
                rmask_dist = abs(rp_tp - rt_tp) * 180 / config.angle_res
                rmask_dist[rmask_dist >= 90] -= 180
                rmask_dist = np.mean(np.abs(rmask_dist))

        else:
            positives = np.logical_and(pred_labels == 1, true_labels == 1)

            rp = np.argmax(rmask_preds.data.cpu().numpy(), axis=2).flatten()
            rt = rmask_targets.data.cpu().numpy().flatten()

            rp_tp = rp[positives]
            rt_tp = rt[positives]

            rot_free = rt_tp == 0

            rp_tp = rp_tp[~rot_free]
            rt_tp = rt_tp[~rot_free]

            if len(rp_tp) == 0:
                rmask_dist = -1
            else:
                rp_tp -= 1
                rt_tp -= 1
                rmask_dist = abs(rp_tp - rt_tp) * 180 / config.angle_res
                rmask_dist[rmask_dist >= 90] -= 180
                rmask_dist = np.mean(np.abs(rmask_dist))

    return bmask_precision, bmask_recall, rmask_dist
