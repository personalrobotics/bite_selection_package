from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPNetLoss(nn.Module):
    def __init__(self):
        super(SPNetLoss, self).__init__()

    def forward(self, pred_positions, gt_positions, pred_angles, gt_angles):
        position_loss = F.smooth_l1_loss(pred_positions, gt_positions)

        I = torch.eye(180)
        pa_onehot = I[pred_angles.round()]
        ga_onehot = I[gt_angles.round()]
        angle_loss = F.cross_entropy(pa_onehot, ga_onehot)

        loss = position_loss + angle_loss
        return loss
