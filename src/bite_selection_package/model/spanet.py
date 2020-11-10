from __future__ import print_function
from __future__ import division

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from bite_selection_package.config import spanet_config as config


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


class DenseSPANet(nn.Module):
    """Densenet-BC model class, based on
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

    # Pretrained block configs:
    # densenet121 (6, 12, 24, 16)
    # densenet169 (6, 12, 32, 32)
    # densenet201 (6, 12, 48, 32)
    # densenet161 (6, 12, 36, 24)
    def __init__(self, growth_rate=32, block_config=[3, 6, 12],
                 num_init_features=64, bn_size=4, drop_rate=0.2,
                 final_vector_size=10, use_rgb=True, use_depth=False):
        super(DenseSPANet, self).__init__()

        self.use_rgb = use_rgb
        self.use_depth = use_depth

        # First convolution
        self.features_rgb = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features,
                kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        self._feature_layers(
            self.features_rgb, num_init_features, block_config, bn_size,
            growth_rate, drop_rate)

        self.features_depth = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_features,
                kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        num_features = self._feature_layers(
            self.features_depth, num_init_features, block_config, bn_size,
            growth_rate, drop_rate)

        self.conv_merge = nn.Sequential(OrderedDict([
            ('merge_conv', nn.Conv2d(
                num_features * 2, num_features, 3, padding=1)),
            ('merge_norm', nn.BatchNorm2d(num_features)),
            ('merge_relu', nn.ReLU(inplace=True)),
        ]))

        # Linear layer
        self.final = nn.Linear(num_features, final_vector_size)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _feature_layers(self, features, num_features, block_config,
                        bn_size, growth_rate, drop_rate):
        # Each denseblock
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers, num_input_features=num_features,
                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2)
                features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        features.add_module('norm5', nn.BatchNorm2d(num_features))
        features.add_module('relu5', nn.ReLU(inplace=True)),
        return num_features

    def forward(self, rgb, depth):
        out_rgb, out_depth = None, None
        if self.use_rgb:
            out_rgb = self.features_rgb(rgb)
        if self.use_depth:
            out_depth = self.features_depth(depth)

        if self.use_rgb and self.use_depth:
            merged = torch.cat((out_rgb, out_depth), 1)
            out = self.conv_merge(merged)
        else:
            out = out_rgb if out_rgb is not None else out_depth

        feat_map = out.clone().detach()
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
        out = self.final(out)
        return out, feat_map

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


class SPANet(nn.Module):
    def __init__(self, final_vector_size=11,
                 use_rgb=True, use_depth=False, use_wall=True):
        super(SPANet, self).__init__()

        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.use_wall = use_wall

        self.conv_init_rgb = nn.Sequential(
            nn.Conv2d(3, 16, 7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),  # 144
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        self.conv_init_depth = nn.Sequential(
            nn.Conv2d(1, 16, 11, padding=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),  # 144
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        self.conv_merge = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv_layers_top = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),  # 72
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),  # 36
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),  # 18
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        self.conv_layers_bot = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),  # 9
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        n_features = 2048
        n_features_final = 2048
        if config.n_features is not None:
            n_features_final = config.n_features

        if self.use_wall:
            n_flattened = 9 * 9 * 256 + 3
        else:
            n_flattened = 9 * 9 * 256

        self.linear_layers = nn.Sequential(
            nn.Linear(n_flattened, n_features),
            nn.BatchNorm1d(n_features),
            nn.ReLU(),
            nn.Linear(n_features, n_features_final),
            nn.BatchNorm1d(n_features_final),
            nn.ReLU(),
        )

        self.final = nn.Linear(n_features_final, final_vector_size)

    def forward(self, rgb, depth, loc_type=None):
        out_rgb, out_depth = None, None
        if self.use_rgb:
            out_rgb = self.conv_init_rgb(rgb)
        if self.use_depth:
            out_depth = self.conv_init_depth(depth)

        if self.use_rgb and self.use_depth:
            merged = torch.cat((out_rgb, out_depth), 1)
            out = self.conv_merge(merged)
        else:
            out = out_rgb if out_rgb is not None else out_depth

        out = self.conv_layers_top(out)
        for _ in range(3):
            out = self.conv_layers_bot(out) + out

        out = out.view(-1, 9 * 9 * 256)

        # Add Wall Detector
        if loc_type is None:
            loc_type = torch.tensor([[1., 0.]]).repeat(out.size()[0], 1) # Isolated = default
            if out.is_cuda:
                loc_type = loc_type.cuda()

        if self.use_wall:
            out = torch.cat((out, loc_type), dim=1)

        out = self.linear_layers(out)
        features = out.clone().detach()

        out = self.final(out)

        out = out.sigmoid()
        return out, features

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
