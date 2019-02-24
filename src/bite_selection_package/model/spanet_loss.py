from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os

import torch.nn as nn

from bite_selection_package.config import spanet_config as config


class SPANetLoss(nn.Module):
    def __init__(self):
        super(SPANetLoss, self).__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_vector, gt_vector):
        pred_vector = pred_vector.view(-1, config.final_vector_size)
        gt_vector = gt_vector.view(-1, config.final_vector_size)

        loss = self.smooth_l1_loss(pred_vector, gt_vector)
        return loss
