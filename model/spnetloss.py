from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPNetLoss(nn.Module):
    def __init__(self):
        super(SPNetLoss, self).__init__()

        self.cel = nn.CrossEntropyLoss()

    def forward(self, pred_positions, gt_positions, pred_angles, gt_angles):
        position_loss = self.cel(
            pred_positions, gt_positions.round().long().view(-1))

        angle_loss = self.cel(
            pred_angles, gt_angles.round().long().view(-1))

        loss = position_loss + angle_loss
        return loss, position_loss, angle_loss
