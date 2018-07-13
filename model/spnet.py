from __future__ import print_function
from __future__ import division

import torch.nn as nn
import torch.nn.functional as F


class SPNet(nn.Module):
    def __init__(self):
        super(SPNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)    # 56
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)   # 28
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)  # 14
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)  # 7
        self.conv4_bn = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 7 * 7, 1024)

        self.fc_pos1 = nn.Linear(1024, 1024)
        self.fc_pos2 = nn.Linear(1024, 1024)
        self.fc_pos3 = nn.Linear(1024, 2)

        self.fc_rot1 = nn.Linear(1024, 1024)
        self.fc_rot2 = nn.Linear(1024, 1024)
        self.fc_rot3 = nn.Linear(1024, 18)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_bn(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.conv4_bn(self.conv4(x)), 2))

        x = x.view(-1, self.num_flat_features(x))
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

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
