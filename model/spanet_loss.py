from __future__ import print_function
from __future__ import division

import sys
import os

import torch.nn as nn

sys.path.append(os.path.split(os.getcwd())[0])
from config import spanet_config as config


class SPANetLoss(nn.Module):
    def __init__(self):
        super(SPANetLoss, self).__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss()

    def forward(self, pred_vector, gt_vector):
        pred_vector = pred_vector.view(-1, config.final_vector_size)
        gt_vector = gt_vector.view(-1, config.final_vector_size)

        loss = self.smooth_l1_loss(pred_vector, gt_vector)
        return loss
