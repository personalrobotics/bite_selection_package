from __future__ import print_function
from __future__ import division

import torch.nn as nn
import torch.nn.functional as F


class SPNet(nn.Module):
    def __init__(self):
        super(SPNet, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),  # 80

            nn.Conv2d(8, 16, 3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),
            nn.ReLU(),  # 40

            nn.Conv2d(16, 32, 3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.ReLU(),  # 20

            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # 10

            nn.Conv2d(64, 128, 3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.ReLU(),  # 5
        )

        self.num_flat_features = 128 * 5 * 5
        self.fc_size = 1024
        self.fc1 = nn.Linear(self.num_flat_features, self.fc_size)

        self.fc_pos = nn.Sequential(
            nn.Linear(self.fc_size, self.fc_size),
            nn.ReLU(),
            nn.Linear(self.fc_size, self.fc_size // 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.fc_size // 2, 2),
        )

        self.fc_rot = nn.Sequential(
            nn.Linear(self.fc_size, self.fc_size),
            nn.ReLU(),
            nn.Linear(self.fc_size, self.fc_size // 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.fc_size // 2, 1),  # 18),
        )

    def forward(self, x):
        x = self.conv_layers(x)

        x = x.view(-1, self.num_flat_features)
        x = F.relu(self.fc1(x))

        pos = self.fc_pos(x)
        rot = self.fc_rot(x)
        return pos, rot
