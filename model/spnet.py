from __future__ import print_function
from __future__ import division

import torch.nn as nn
import torch.nn.functional as F


class SPNet(nn.Module):
    def __init__(self):
        super(SPNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 8, 3)    # 56
        self.conv2 = nn.Conv2d(8, 16, 3)   # 28
        self.conv3 = nn.Conv2d(16, 32, 3)  # 14
        self.conv4 = nn.Conv2d(32, 64, 3)  # 7

        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
