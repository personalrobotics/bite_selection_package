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


class SPNetLoss(nn.Module):
    def __init__(self):
        super(SPNetLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(size_average=True)
        self.ce_loss = nn.CrossEntropyLoss(size_average=True)

    def forward(self, bmask_preds, bmask_targets, rmask_preds, rmask_targets):
        bmask_preds = bmask_preds.view(-1, config.mask_size ** 2)
        bmask_targets = bmask_targets.view(-1, config.mask_size ** 2)

        positives = bmask_targets > 0

        bmask_preds_pos = bmask_preds[positives]
        bmask_targets_pos = bmask_targets[positives]
        bmask_loss_pos = self.bce_loss(bmask_preds_pos, bmask_targets_pos)

        bmask_preds_neg = bmask_preds[~positives]
        bmask_targets_neg = bmask_targets[~positives]
        bmask_loss_neg = self.bce_loss(bmask_preds_neg, bmask_targets_neg)

        pw = 0.3
        bmask_loss = pw * bmask_loss_pos + (1 - pw) * bmask_loss_neg

        if config.use_rotation:
            if config.use_rot_alt:
                rmask_preds = rmask_preds.view(-1, config.mask_size ** 2)
                rmask_targets = rmask_targets.view(-1, config.mask_size ** 2)
                rmask_loss = torch.mean(torch.sin(
                    (torch.abs(rmask_targets_full - rmask_preds_full)
                     * 3.141592 / 180)))
            else:
                rmask_targets = rmask_targets.view(-1, config.mask_size ** 2)
                mask = positives.unsqueeze(2).expand_as(rmask_preds)

                # rmask_preds_pos = rmask_preds[mask].view(-1, config.angle_res + 1)
                # rmask_targets_pos = rmask_targets[positives].view(-1)
                rmask_preds_pos = rmask_preds[mask].view(-1, config.angle_res + 1)
                rmask_targets_pos = rmask_targets[positives]
                rot_pos = rmask_targets_pos > 0
                rmask_targets_pos = rmask_targets_pos[rot_pos]
                rmask_preds_pos = rmask_preds_pos[
                    rot_pos.unsqueeze(1).expand_as(rmask_preds_pos)]
                rmask_preds_pos = rmask_preds_pos.view(-1, config.angle_res + 1)
                rmask_targets_pos = rmask_targets_pos.view(-1)
                if rmask_targets_pos.size() > torch.Size([0]):
                    rmask_loss_pos = self.ce_loss(
                        rmask_preds_pos, rmask_targets_pos.long())
                else:
                    rmask_loss_pos = 0

                rmask_preds_neg = rmask_preds[~mask].view(-1, config.angle_res + 1)
                rmask_targets_neg = rmask_targets[~positives].view(-1)
                rmask_loss_neg = self.ce_loss(
                    rmask_preds_neg, rmask_targets_neg.long())

                pw = 0.9
                rmask_loss = pw * rmask_loss_pos + (1 - pw) * rmask_loss_neg
        else:
            rmask_loss_full = 0
            rmask_loss = 0

        loss = bmask_loss + rmask_loss

        return loss, bmask_loss, rmask_loss
