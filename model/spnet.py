from __future__ import print_function
from __future__ import division

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from config import config


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseSPNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0):
        super(DenseSPNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        block_config = (3, 3)
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final masks
        self.mask_ch = 128
        self.final_conv = nn.Conv2d(176, self.mask_ch, 1, padding=0)

        self.final_layers_bin = nn.Sequential(
            nn.Conv2d(self.mask_ch, self.mask_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.mask_ch, self.mask_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.mask_ch, self.mask_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.mask_ch, 1, 1, padding=0),
        )

        self.final_layers_rot = nn.Sequential(
            nn.Conv2d(self.mask_ch, self.mask_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.mask_ch, self.mask_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.mask_ch, self.mask_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.mask_ch, config.angle_res + 1, 1, padding=0),
        )

        # Final batch norm
        # self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        # self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        # out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        # out = self.classifier(out)

        out = F.relu(self.final_conv(out))

        bmask = self.final_layers_bin(out)
        bmask = bmask.permute(0, 2, 3, 1).contiguous().view(
            out.size(0), config.mask_size ** 2)

        rmask = self.final_layers_rot(out)
        rmask = rmask.permute(0, 2, 3, 1).contiguous().view(
            out.size(0), config.mask_size ** 2, config.angle_res + 1)

        return bmask, rmask

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


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
            nn.ReLU(),

            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        self.conv_layers_bot = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
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

        if config.use_rot_alt:
            self.final_layers_rot_alt = nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 1, 1, padding=0),
            )

    def forward(self, x):
        x = self.conv_layers_top(x)

        for _ in range(3):
            x = self.conv_layers_bot(x)

        bmask = self.final_layers_bin(x)
        bmask = bmask.permute(0, 2, 3, 1).contiguous().view(
            x.size(0), config.mask_size ** 2)

        if config.use_rot_alt:
            rmask = self.final_layers_rot_alt(x)
            rmask = rmask.permute(0, 2, 3, 1).contiguous().view(
                x.size(0), config.mask_size ** 2)
        else:
            rmask = self.final_layers_rot(x)
            rmask = rmask.permute(0, 2, 3, 1).contiguous().view(
                x.size(0), config.mask_size ** 2, config.angle_res + 1)

        return bmask, rmask

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
