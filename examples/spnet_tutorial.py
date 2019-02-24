#!/usr/bin/env python

import torch
import torchvision.transforms as transforms
from PIL import Image
from bite_selection_package.model.spnet import DenseSPNet


def spnet_tutorial():
    spnet = DenseSPNet()
    checkpoint = torch.load('../checkpoint/spnet_ckpt.pth')
    spnet.load_state_dict(checkpoint['net'])
    spnet.eval()
    img = Image.open('./sample.jpg')
    img = img.resize((136, 136))
    trans = transforms.ToTensor()
    inp = torch.stack([trans(img)])

    pred_bmasks, pred_rmasks = spnet(inp)
    print('binary mask: ')
    print(pred_bmasks.data)
    print('rotation mask: ')
    print(pred_rmasks.data)


if __name__ == '__main__':
    spnet_tutorial()
