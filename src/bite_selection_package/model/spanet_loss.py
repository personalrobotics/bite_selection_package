from __future__ import print_function
from __future__ import division

import sys
import os

import torch.nn as nn


class SPANetLoss(nn.Module):
    def __init__(self, final_vector_size=10):
        super(SPANetLoss, self).__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.final_vector_size = final_vector_size

    def forward(self, pred_vector, gt_vector):
        pred_vector = pred_vector.view(-1, self.final_vector_size)
        gt_vector = gt_vector.view(-1, self.final_vector_size)

        loss = self.smooth_l1_loss(pred_vector, gt_vector)
        return loss
