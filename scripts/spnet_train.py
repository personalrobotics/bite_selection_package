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


os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def train_spnet(use_cuda):
    print('train_spnet')
    print('use cuda: ' + str(use_cuda))

    img_base_dir = '../data/processed/cropped_images/'

    train_list = '../data/processed/sp_train.txt'
    test_list = '../data/processed/sp_test.txt'

    checkpoint_dir = '../checkpoints/'
    checkpoint_path = os.path.join(checkpoint_dir, 'spnet_ckpt.pth')

    sample_dir = '../samples/'
    img_dir = sample_dir + 'cropped_images/'
    ann_dir = sample_dir + 'annotations/'
    if os.path.exists(sample_dir):
        shutil.rmtree(sample_dir)
    os.makedirs(sample_dir)
    os.makedirs(img_dir)
    os.makedirs(ann_dir)

    transform = transforms.Compose([
        transforms.ToTensor()])
        # transforms.Normalize((0.562, 0.370, 0.271), (0.332, 0.302, 0.281))])

    trainset = SPDataset(
        root=img_base_dir,
        list_file=train_list,
        train=True, transform=transform, img_size=56)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=8,
        shuffle=True, num_workers=8,
        collate_fn=trainset.collate_fn)

    testset = SPDataset(
        root=img_base_dir,
        list_file=test_list,
        train=False, transform=transform, img_size=56)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4,
        shuffle=False, num_workers=8,
        collate_fn=testset.collate_fn)

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

    for epoch in range(start_epoch, start_epoch + 1000):
        # training
        print('\nEpoch: {}'.format(epoch))
        spnet.train()
        train_loss = 0

        total_batches = int(math.ceil(
            trainloader.dataset.num_samples / trainloader.batch_size))

        for batch_idx, (imgs, positions, angles) in enumerate(trainloader):
            if use_cuda:
                imgs = imgs.cuda()
                gt_positions = positions.cuda()
                gt_angles = angles.cuda()
            else:
                gt_positions = positions
                gt_angles = angles

            optimizer.zero_grad()
            pred_positions, pred_angles = spnet(imgs)

            loss, ploss, rloss = criterion(
                pred_positions, gt_positions, pred_angles, gt_angles)
            loss.backward()
            optimizer.step()

            train_loss += loss.data

            if batch_idx % 20 == 0:
                print('[{0}| {1:3d}/{2:3d}] train_loss: {3:6.3f} '
                      '(p={5:.3f}, r={6:.3f}) | avg_loss: {4:6.3f}'.format(
                    epoch, batch_idx, total_batches,
                    loss.data, train_loss / (batch_idx + 1),
                    ploss, rloss))

        # testing
        print('\nTest')
        spnet.eval()
        test_loss = 0

        total_batches = int(math.ceil(
            testloader.dataset.num_samples / testloader.batch_size))

        for batch_idx, (imgs, positions, angles) in enumerate(testloader):
            if use_cuda:
                imgs = imgs.cuda()
                gt_positions = positions.cuda()
                gt_angles = angles.cuda()
            else:
                gt_positions = positions
                gt_angles = angles

            pred_positions, pred_angles = spnet(imgs)
            loss, ploss, rloss = criterion(
                pred_positions, gt_positions, pred_angles, gt_angles)

            test_loss += loss.data

            if batch_idx % 2 == 0:
                print('[{0}| {1:3d}/{2:3d}]  test_loss: {3:6.3f} '
                      '(p={5:.3f}, r={6:.3f}) | avg_loss: {4:6.3f}'.format(
                    epoch, batch_idx, total_batches,
                    loss.data, test_loss / (batch_idx + 1),
                    ploss, rloss))

                # save a sample prediction
                ann_path_base = os.path.join(
                    ann_dir, 'test_{0:04d}_{1:04d}'.format(
                        epoch, batch_idx))
                img_path_base = os.path.join(
                    img_dir, 'test_{0:04d}_{1:04d}'.format(
                        epoch, batch_idx))
                sample_img_path = img_path_base + '.jpg'
                sample_data_path = ann_path_base + '.out'

                test_img = imgs[0].cpu()
                utils.save_image(test_img, sample_img_path)
                pred_posdata = pred_positions[0].data.cpu().numpy()
                pred_posnumber = np.argmax(pred_posdata)
                predx = (pred_posnumber % 8) * 7
                predy = (pred_posnumber / 8) * 7
                pred_pos = [predx, predy]
                pred_ang = pred_angles[0].data.cpu().numpy()
                pred_ang = np.argmax(pred_ang) * 10

                with open(sample_data_path, 'w') as f:
                    f.write('{0:.3f} {1:.3f} {2:.3f}'.format(
                        pred_pos[0], pred_pos[1], pred_ang))
                    f.close()

        # save checkpoint
        test_loss /= len(testloader)
        if test_loss < best_loss:
            print('Saving checkpoint..')
            state = {
                'net': spnet.module.state_dict(),
                'loss': test_loss,
                'epoch': epoch, }
            if not os.path.exists(os.path.dirname(checkpoint_path)):
                os.makedirs(os.path.dirname(checkpoint_path))
            torch.save(state, checkpoint_path)
            best_loss = test_loss


if __name__ == '__main__':
    if len(sys.argv) == 2:
        train_spnet(sys.argv[1] != 'nocuda')
    else:
        train_spnet(True)

