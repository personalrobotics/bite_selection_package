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

        self.fc_pos1 = nn.Linear(self.fc_size, self.fc_size)
        self.fc_pos2 = nn.Linear(self.fc_size, self.fc_size)
        self.fc_pos3 = nn.Linear(self.fc_size, 2)

        self.fc_rot1 = nn.Linear(self.fc_size, self.fc_size)
        self.fc_rot2 = nn.Linear(self.fc_size, self.fc_size)
        self.fc_rot3 = nn.Linear(self.fc_size, 18)

    def forward(self, x):
        x = self.conv_layers(x)

        x = x.view(-1, self.num_flat_features)
        x = F.relu(self.fc1(x))

        pos = F.relu(self.fc_pos1(x))
        pos = F.relu(self.fc_pos2(pos))
        pos = F.dropout(pos, training=self.training)
        pos = self.fc_pos3(pos)

        rot = F.relu(self.fc_rot1(x))
        rot = F.relu(self.fc_rot2(rot))
        rot = F.dropout(rot, training=self.training)
        rot = self.fc_rot3(rot)
        return pos, rot
