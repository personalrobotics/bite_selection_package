from __future__ import print_function
from __future__ import division

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPNetLoss(nn.Module):
    def __init__(self):
        super(SPNetLoss, self).__init__()
        self.sll = nn.SmoothL1Loss()
        #self.cel = nn.CrossEntropyLoss()

    def forward(self, pred_positions, gt_positions, pred_angles, gt_angles):
        position_loss = self.sll(pred_positions, gt_positions)

        #angle_loss = self.cel(pred_angles, gt_angles.round().long().view(-1))
        angle_loss = torch.mean(torch.abs(
            torch.sin((torch.abs(pred_angles - gt_angles) / 180 * math.pi))))

        loss = position_loss + angle_loss

        return loss, position_loss, angle_loss
