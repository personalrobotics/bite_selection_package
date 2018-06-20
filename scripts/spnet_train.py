#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import os
import sys
import math

import torch
import torch.optim as optim
import torchvision.transforms as transforms

sys.path.append('../')

from model.spnet import SPNet
from model.spdataset import SPDataset
from model.spnetloss import SPNetLoss


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train_spnet():
    print('train_spnet')

    img_base_dir = '../data/processed/cropped_images/'

    train_list = '../data/processed/sp_train.txt'
    test_list = '../data/processed/sp_test.txt'

    checkpoint_path = '../checkpoints/spnet_ckpt.pth'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))])

    trainset = SPDataset(
        root=img_base_dir,
        list_file=train_list,
        train=True, transform=transform, input_size=600)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=8,
        shuffle=True, num_workers=8,
        collate_fn=trainset.collate_fn)

    testset = SPDataset(
        root=img_base_dir,
        list_file=test_list,
        train=False, transform=transform, input_size=600)
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
    spnet = spnet.cuda()

    criterion = SPNetLoss()
    optimizer = optim.SGD(spnet.parameters(), lr=1e-3,
                          momentum=0.9, weight_decay=1e-4)

    for epoch in range(start_epoch, start_epoch + 100):
        # training
        print('\nEpoch: {}'.format(epoch))
        spnet.train()
        train_loss = 0

        total_batches = int(math.ceil(
            trainloader.dataset.num_samples / trainloader.batch_size))

        for batch_idx, (imgs, positions, angles) in enumerate(trainloader):
            imgs = imgs.cuda()
            gt_positions = positions.cuda()
            gt_angles = angles.cuda()

            optimizer.zero_grad()
            pred_positions, pred_angles = spnet(imgs)
            loss, ploss, rloss = criterion(
                pred_positions, gt_positions, pred_angles, gt_angles)
            loss.backward()
            optimizer.step()

            train_loss += loss.data

            if batch_idx % 20 == 0:
                print('[{0}| {1}/{2}] train_loss: {3:.3f} ({5:.3f}, {6:.3f}) | avg_loss: {4:.3f}'.format(
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
            imgs = imgs.cuda()
            gt_positions = positions.cuda()
            gt_angles = angles.cuda()

            pred_positions, pred_angles = spnet(imgs)
            loss, ploss, rloss = criterion(
                pred_positions, gt_positions, pred_angles, gt_angles)

            test_loss += loss.data

            if batch_idx % 10 == 0:
                print('[{0}| {1}/{2}] test_loss: {3:.3f} ({5:.3f}, {6:.3f}) | avg_loss: {4:.3f}'.format(
                    epoch, batch_idx, total_batches,
                    loss.data, test_loss / (batch_idx + 1),
                    ploss, rloss))

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
    train_spnet()
