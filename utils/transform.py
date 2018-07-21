from __future__ import division

import numpy as np
import random

import torch

from PIL import Image, ImageDraw


def resize(img, loc, rot, size):
    w, h = img.size

    if isinstance(size, int):
        size_min = min(w, h)
        size_max = max(w, h)
        sw = sh = float(size) / size_min
        if sw * size_max > max_size:
            sw = sh = float(max_size) / size_max
        ow = int(w * sw + 0.5)
        oh = int(h * sh + 0.5)
    else:
        ow, oh = size
        sw = float(ow) / w
        sh = float(oh) / h

    img = img.resize((ow, oh), Image.BILINEAR)
    loc = loc * torch.Tensor([sw, sh])
    return img, loc, rot


def random_flip(img, loc, rot):
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        loc[0] = 1 - loc[0]
        rot = (18 - rot) % 18
    return img, loc, rot


def random_rotate(img, loc, rot):
    roff = random.random() * 180

    img = img.rotate(roff, resample=Image.BILINEAR)
    loc = list(rotate_point(loc, roff))

    if rot > 0:
        rot = (rot + np.round(float(roff) / 10)) % 18
    return img, loc, rot


def rotate_point(p, angle):
    s = np.sin(np.radians(-angle))
    c = np.cos(np.radians(-angle))
    tx = p[0] - 0.5
    ty = p[1] - 0.5
    nx = tx * c - ty * s
    ny = tx * s + ty * c
    nx += 0.5
    ny += 0.5
    return nx, ny

