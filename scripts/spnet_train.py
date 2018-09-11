#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import numpy as np
import sys
import os
import shutil
import math

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as utils

sys.path.append(os.path.split(os.getcwd())[0])
from model.spnet import SPNet, DenseSPNet
from model.spdataset import SPDataset
from model.spnetloss import SPNetLoss
from config import config


os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id


def train_spnet(use_cuda=True):
    print('train_spnet')
    print('use cuda: {}'.format(use_cuda))
    print('use identity: {}'.format(config.use_identity))
    print('use rotation: {}'.format(config.use_rotation))
    print('use rot_alt: {}'.format(config.use_rot_alt))
    print('use densenet: {}'.format(config.use_densenet))

    img_base_dir = config.cropped_img_dir

    train_list = config.train_list_filename
    test_list = config.test_list_filename

    checkpoint_path = config.checkpoint_filename
    checkpoint_path_best = config.checkpoint_best_filename

    sample_dir_name = os.path.join('samples', config.project_prefix)
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
    trainset = SPDataset(
        cropped_img_dir=config.cropped_img_dir,
        mask_dir=config.mask_dir,
        list_filename=config.train_list_filename,
        label_map_filename=config.label_map_filename,
        train=True,
        transform=transform,
        cropped_img_res=config.cropped_img_res)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.train_batch_size,
        shuffle=True, num_workers=8,
        collate_fn=trainset.collate_fn)

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

    if config.use_densenet:
        spnet = DenseSPNet()
    else:
        spnet = SPNet()
    print(spnet)

    best_loss = float('inf')
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        print('Resuming from checkpoint \"{}\"'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        spnet.load_state_dict(checkpoint['net'])
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']

    spnet = torch.nn.DataParallel(
        spnet, device_ids=range(torch.cuda.device_count()))
    if use_cuda:
        spnet = spnet.cuda()

    criterion = SPNetLoss()
    optimizer = optim.SGD(spnet.parameters(), lr=1e-3,
                          momentum=0.9, weight_decay=1e-4)

    print(trainloader.dataset.num_samples)
    print(testloader.dataset.num_samples)

    for epoch in range(start_epoch, start_epoch + 10000):
        # training
        print('\nEpoch: {}'.format(epoch))
        spnet.train()
        # spnet.module.freeze_bn()
        train_loss = 0

        total_batches = int(math.ceil(
            trainloader.dataset.num_samples / trainloader.batch_size))

        for batch_idx, batch_items in enumerate(trainloader):
            imgs = batch_items[0]
            gt_bmasks = batch_items[1]
            gt_rmasks = batch_items[2]

            if use_cuda:
                imgs = imgs.cuda()
                gt_bmasks = gt_bmasks.cuda()
                gt_rmasks = gt_rmasks.cuda()

            optimizer.zero_grad()
            pred_bmasks, pred_rmasks = spnet(imgs)

            loss, bmloss, rmloss = criterion(
                pred_bmasks, gt_bmasks, pred_rmasks, gt_rmasks)
            loss.backward()
            optimizer.step()

            train_loss += loss.data

            if batch_idx % 20 == 0:
                print('[{0}| {1:3d}/{2:3d}] train_loss: {3:6.3f} '
                      '(b={4:.3f}, r={5:.3f}) | avg_loss: {6:6.3f}'.format(
                    epoch, batch_idx, total_batches, loss.data,
                    bmloss, rmloss, train_loss / (batch_idx + 1)))

                # # save a sample ground truth data
                # ann_path_base = os.path.join(
                #     ann_dir, 'train_{0:04d}_{1:04d}'.format(
                #         epoch, batch_idx))
                # img_path_base = os.path.join(
                #     img_dir, 'train_{0:04d}_{1:04d}'.format(
                #         epoch, batch_idx))
                # sample_img_path = img_path_base + '.jpg'
                # sample_data_path = ann_path_base + '.out'

                # test_img = imgs[0].cpu()
                # utils.save_image(test_img, sample_img_path)
                # gt_pos = gt_positions[0].data.cpu().numpy()
                # gt_ang = gt_angles[0].data.cpu().numpy()[0] * 10

                # with open(sample_data_path, 'w') as f:
                #     f.write('{0:.3f} {1:.3f} {2:.3f}'.format(
                #         gt_pos[0], gt_pos[1], gt_ang))
                #     f.close()

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

            if batch_idx % 20 == 0:
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

                negatives = pred_bmasks[0].data.cpu().numpy() < 0
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

        # save checkpoint
        state = {
            'net': spnet.module.state_dict(),
            'loss': test_loss,
            'epoch': epoch, }
        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state, checkpoint_path)

        test_loss /= len(testloader)
        if test_loss < best_loss:
            print('Saving best checkpoint..')
            state = {
                'net': spnet.module.state_dict(),
                'loss': test_loss,
                'epoch': epoch, }
            if not os.path.exists(os.path.dirname(checkpoint_path_best)):
                os.makedirs(os.path.dirname(checkpoint_path_best))
            torch.save(state, checkpoint_path_best)
            best_loss = test_loss


if __name__ == '__main__':
    if len(sys.argv) == 2:
        train_spnet(sys.argv[1] != 'nocuda')
    else:
        train_spnet()

