#!/usr/bin/env python

import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

from PIL import Image
from bite_selection_package.model.spnet import DenseSPNet


def spnet_tutorial():
    # initialize DenseSPNet and load the trained checkpoint
    spnet = DenseSPNet()
    checkpoint = torch.load('../checkpoint/spnet_ckpt.pth')
    spnet.load_state_dict(checkpoint['net'])

    # prepare an input image
    img = Image.open('./sample.jpg')
    img = img.resize((136, 136))
    trans = transforms.ToTensor()
    inp = torch.stack([trans(img)])

    # get predictions
    spnet.eval()
    pred_bmasks, pred_rmasks = spnet(inp)

    # print and visualize predicted masks
    print('rotation mask: ')
    print(pred_rmasks.data)

    print('visualize binary mask for skewering locations')
    bm_arr = pred_bmasks.data[0].sigmoid().view(17, 17).numpy()

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(bm_arr)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    spnet_tutorial()
