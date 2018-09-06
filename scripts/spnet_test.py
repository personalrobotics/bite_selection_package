#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import numpy as np
import os
import shutil
import sys
import math

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as utils

sys.path.append('../')

from model.spnet import SPNet
from model.spdataset import SPDataset
from model.spnetloss import SPNetLoss
from config import config


os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id


def test_spnet(use_cuda=True):
    print('test_spnet')
    print('use cuda: {}'.format(use_cuda))
    print('use identity: {}'.format(config.use_identity))
    print('use rotation: {}'.format(config.use_rotation))
    print('use rot_alt: {}'.format(config.use_rot_alt))

    img_base_dir = config.cropped_img_dir

    test_list = config.test_list_filename

    checkpoint_path = config.checkpoint_filename
    checkpoint_path_best = config.checkpoint_best_filename

    sample_dir_name = os.path.join('test_rst', config.project_prefix)
    sample_dir = os.path.join(
        config.project_dir, sample_dir_name)

    img_dir = os.path.join(sample_dir, 'cropped_images')
    ann_dir = os.path.join(sample_dir, 'masks')
    if os.path.exists(sample_dir):
        shutil.rmtree(sample_dir)
    os.makedirs(sample_dir)
    os.makedirs(img_dir)
    os.makedirs(ann_dir)

    img_size = config.cropped_img_res

    transform = transforms.Compose([
        transforms.ToTensor()])
        # transforms.Normalize((0.562, 0.370, 0.271), (0.332, 0.302, 0.281))])

    print('load SPDataset')
    testset = SPDataset(
        cropped_img_dir=config.cropped_img_dir,
        mask_dir=config.mask_dir,
        list_filename=config.test_list_filename,
        label_map_filename=config.label_map_filename,
        train=False,
        transform=transform,
        cropped_img_res=config.cropped_img_res)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config.test_batch_size,
        shuffle=True, num_workers=8,
        collate_fn=testset.collate_fn)

    spnet = SPNet()
    print(spnet)

    if not os.path.exists(checkpoint_path_best):
        print('Cannot find checkpoint')
        return

    print('Load checkpoint \"{}\"'.format(checkpoint_path_best))
    checkpoint = torch.load(checkpoint_path_best)
    spnet.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    epoch = checkpoint['epoch']

    spnet = torch.nn.DataParallel(
        spnet, device_ids=range(torch.cuda.device_count()))
    if use_cuda:
        spnet = spnet.cuda()

    criterion = SPNetLoss()

    print(testloader.dataset.num_samples)

    # testing
    print('\nTest')
    spnet.eval()
    test_loss = 0

    total_batches = int(math.ceil(
        testloader.dataset.num_samples / testloader.batch_size))

    for batch_idx, batch_items in enumerate(testloader):
        imgs = batch_items[0]
        gt_bmasks = batch_items[1]
        gt_rmasks = batch_items[2]

        if use_cuda:
            imgs = imgs.cuda()
            gt_bmasks = gt_bmasks.cuda()
            gt_rmasks = gt_rmasks.cuda()

        pred_bmasks, pred_rmasks = spnet(imgs)
        loss, bmloss, rmloss = criterion(
            pred_bmasks, gt_bmasks, pred_rmasks, gt_rmasks)

        test_loss += loss.data

        if batch_idx % 10 == 0:
            print('[{0}| {1:3d}/{2:3d}] test_loss: {3:6.3f} '
                  '(b={4:.3f}, r={5:.3f}) | avg_loss: {6:6.3f}'.format(
                epoch, batch_idx, total_batches, loss.data,
                bmloss, rmloss, test_loss / (batch_idx + 1)))

            # save a sample prediction
            img_path_base = os.path.join(
                img_dir, 'test_{0:04d}_apple_{1:04d}'.format(
                    epoch, batch_idx))
            mask_path_base = os.path.join(
                ann_dir, 'test_{0:04d}_apple_{1:04d}'.format(
                    epoch, batch_idx))
            sample_img_path = img_path_base + '.jpg'
            sample_mask_path = mask_path_base + '.txt'

            test_img = imgs[0].cpu()
            utils.save_image(test_img[:3], sample_img_path)

            negatives = pred_bmasks[0].data.cpu().numpy() < -1
            rmask = pred_rmasks[0].data.cpu().numpy()
            rmask = np.argmax(rmask, axis=1) - 1
            rmask = rmask * 180 / config.angle_res
            rmask[rmask < 0] = 0

            rmask[negatives] = -1
            rmask = rmask.reshape(config.mask_size, config.mask_size)

            with open(sample_mask_path, 'w') as f:
                for ri in range(config.mask_size):
                    for ci in range(config.mask_size):
                        f.write('{0:.1f}'.format(rmask[ri][ci]))
                        if ci < config.mask_size - 1:
                            f.write(',')
                    f.write('\n')
                f.close()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        test_spnet(sys.argv[1] != 'nocuda')
    else:
        test_spnet()

