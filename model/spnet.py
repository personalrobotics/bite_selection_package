from __future__ import print_function
from __future__ import division

import sys

import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../')
from config import config


class SPNet(nn.Module):
    def __init__(self):
        super(SPNet, self).__init__()

        if config.use_identity:
            input_channels = 4
        else:
            input_channels = 3

        self.conv_layers_top = nn.Sequential(
            nn.Conv2d(input_channels, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),  # 72

            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.ReLU(),  # 36

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),  # 18

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),  # 9
        )

        self.conv_layers_bot = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),  # 17
        )

        self.final_layers_bin = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1, padding=0),
        )

        self.final_layers_rot = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, config.angle_res + 1, 1, padding=0),
        )

        # self.num_flat_features = 64 * config.mask_size ** 2
        # self.fc_size = 1024
        # self.fc1 = nn.Linear(self.num_flat_features, self.fc_size)

        # self.fc_pos = nn.Sequential(
        #     nn.Linear(self.fc_size, self.fc_size),
        #     nn.ReLU(),
        #     nn.Linear(self.fc_size, self.fc_size // 2),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(self.fc_size // 2, 2),
        # )

        # self.fc_rot = nn.Sequential(
        #     nn.Linear(self.fc_size, self.fc_size),
        #     nn.ReLU(),
        #     nn.Linear(self.fc_size, self.fc_size // 2),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(self.fc_size // 2, 1),  # 18),
        # )

    def forward(self, x):
        x = self.conv_layers_top(x)

        for _ in range(3):
            x = self.conv_layers_bot(x)

        bmask = self.final_layers_bin(x)
        rmask = self.final_layers_rot(x)

        bmask = bmask.permute(0, 2, 3, 1).contiguous().view(
            x.size(0), config.mask_size ** 2)

        rmask = rmask.permute(0, 2, 3, 1).contiguous().view(
            x.size(0), config.mask_size ** 2, config.angle_res + 1)

        return bmask, rmask

        # x = x.view(-1, self.num_flat_features)
        # x = F.relu(self.fc1(x))

        # pos = self.fc_pos(x)
        # rot = self.fc_rot(x)
        # return pos, rot
