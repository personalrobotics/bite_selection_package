from __future__ import print_function
from __future__ import division

import math
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from config import config


class SPNetLoss(nn.Module):
    def __init__(self):
        super(SPNetLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, bmask_preds, bmask_targets, rmask_preds, rmask_targets):
        bmask_preds = bmask_preds.view(-1, config.mask_size ** 2)
        bmask_targets = bmask_targets.view(-1, config.mask_size ** 2)
        bmask_loss = self.bce_loss(bmask_preds, bmask_targets)

        if config.use_rotation:
            if config.use_rot_alt:
                rmask_preds_full = rmask_preds.view(-1, config.angle_res + 1)
                rmask_targets_full = rmask_targets.view(-1)
                rmask_loss_full = self.ce_loss(
                    rmask_preds_full, rmask_targets_full.long())

                positives = bmask_targets > 0
                mask = positives.unsqueeze(2).expand_as(rmask_preds)
                rmask_preds = rmask_preds[mask].view(-1, config.angle_res + 1)
                rmask_targets = rmask_targets[positives]
                rmask_loss = self.ce_loss(rmask_preds, rmask_targets.long())
            else:
                rmask_preds_full = rmask_preds.view(-1, config.angle_res + 1)
                rmask_targets_full = rmask_targets.view(-1)
                rmask_loss_full = self.ce_loss(
                    rmask_preds_full, rmask_targets_full.long())
                rmask_loss_full *= 1e-1

                positives = bmask_targets > 0
                mask = positives.unsqueeze(2).expand_as(rmask_preds)
                rmask_preds = rmask_preds[mask].view(-1, config.angle_res + 1)
                rmask_targets = rmask_targets[positives]
                rmask_loss = self.ce_loss(rmask_preds, rmask_targets.long())
        else:
            rmask_loss_full = 0
            rmask_loss = 0

        loss = bmask_loss + rmask_loss_full + rmask_loss

        return loss, bmask_loss, rmask_loss
