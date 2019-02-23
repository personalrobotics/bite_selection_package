from __future__ import print_function
from __future__ import division

import math
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.split(os.getcwd())[0])
from config import spnet_config as config


class SPNetLoss(nn.Module):
    def __init__(self):
        super(SPNetLoss, self).__init__()

    def forward(self, bmask_preds, bmask_targets, rmask_preds, rmask_targets):
        bmask_preds = bmask_preds.view(-1, config.mask_size ** 2)
        bmask_targets = bmask_targets.view(-1, config.mask_size ** 2)
        positives = bmask_targets == 1

        bmask_loss = F.binary_cross_entropy_with_logits(
            bmask_preds, bmask_targets, weight=config.p_weight)

        if config.use_rotation:
            rmask_targets = rmask_targets.view(-1, config.mask_size ** 2)
            mask = positives.unsqueeze(2).expand_as(rmask_preds)

            rmask_preds_pos = rmask_preds[mask].view(-1, config.angle_res + 1)
            rmask_targets_pos = rmask_targets[positives].view(-1)
            if rmask_targets_pos.size() > torch.Size([0]):
                rmask_loss_pos = F.cross_entropy(
                    rmask_preds_pos, rmask_targets_pos.long())
            else:
                rmask_loss_pos = 0

            rmask_preds_neg = rmask_preds[~mask].view(-1, config.angle_res + 1)
            rmask_targets_neg = rmask_targets[~positives].view(-1)
            rmask_loss_neg = F.cross_entropy(
                rmask_preds_neg, rmask_targets_neg.long())

            pw = 0.99
            rmask_loss = pw * rmask_loss_pos + (1 - pw) * rmask_loss_neg
        else:
            rmask_loss_full = 0
            rmask_loss = 0

        loss = bmask_loss + rmask_loss

        return loss, bmask_loss, rmask_loss
