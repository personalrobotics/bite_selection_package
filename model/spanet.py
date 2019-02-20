from __future__ import print_function
from __future__ import division

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

sys.path.append(os.path.split(os.getcwd())[0])
from config import spanet_config as config


class SPANet(nn.Module):
    def __init__(self):
        super(SPANet, self).__init__()

        input_channels = 0
        if config.use_rgb:
            input_channels += 3
        if config.use_depth:
            input_channels += 1

        self.conv_layers_top = nn.Sequential(
            nn.Conv2d(input_channels, 8, 7, padding=3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),  # 144
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),  # 72
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),  # 36
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),  # 18
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        self.conv_layers_bot = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),  # 9
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        n_features = 4096

        self.linear_layers = nn.Sequential(
            nn.Linear(9 * 9 * 128, n_features),
            nn.BatchNorm1d(n_features),
            nn.ReLU(),
            nn.Linear(n_features, n_features),
            nn.BatchNorm1d(n_features),
            nn.ReLU(),
        )

        self.final = nn.Linear(n_features, config.final_vector_size)

    def forward(self, x):
        out = self.conv_layers_top(x)

        for _ in range(3):
            out = self.conv_layers_bot(out) + out

        out = out.view(-1, 9 * 9 * 128)
        out = self.linear_layers(out)
        out = self.final(out)
        return out

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
