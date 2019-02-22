#!/usr/bin/env python3

from __future__ import division
from __future__ import print_function

import sys
import os
import math

import torch
import torch.optim as optim
import torchvision.transforms as transforms

sys.path.append(os.path.split(os.getcwd())[0])
from model.spnet import SPNet, DenseSPNet
from model.spnet_dataset import SPDataset
from model.spnet_loss import SPNetLoss
from config import spnet_config as config
from spnet_utils.utils import get_accuracy


os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id


def train_spnet(use_cuda=config.use_cuda):
    print('train_spnet')
    print('use cuda: {}'.format(use_cuda))
    print('use rotation: {}'.format(config.use_rotation))
    print('use densenet: {}'.format(config.use_densenet))

    transform = transforms.Compose([
        transforms.ToTensor()])
        # transforms.Normalize((0.562, 0.370, 0.271), (0.332, 0.302, 0.281))])

    exp_mode = 'full' if config.excluded_item is None else 'exclude'

    print('load SPDataset')
    trainset = SPDataset(
        list_filename=config.train_list_filename,
        train=True,
        exp_mode=exp_mode,
        transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.train_batch_size,
        shuffle=True, num_workers=8,
        collate_fn=trainset.collate_fn)

    testset = SPDataset(
        list_filename=config.test_list_filename,
        train=False,
        exp_mode=exp_mode,
        transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config.test_batch_size,
        shuffle=True, num_workers=8,
        collate_fn=testset.collate_fn)

    if config.use_densenet:
        spnet = DenseSPNet()
    else:
        spnet = SPNet()

    best_loss = float('inf')
    start_epoch = 0

    checkpoint_path = config.checkpoint_filename
    checkpoint_path_best = config.checkpoint_best_filename

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

    print('training set: {}'.format(trainloader.dataset.num_samples))
    print('test set: {}'.format(testloader.dataset.num_samples))

    for epoch in range(start_epoch, start_epoch + 10000):
        # training
        print('\nEpoch: {} | {}'.format(epoch, config.project_prefix))
        spnet.train()
        # spnet.module.freeze_bn()
        train_loss = 0

        this_bacc = 0
        this_bpre = 0
        this_brec = 0
        this_bf1 = 0
        this_rdist = 0

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

            bacc, bpre, brec, bf1, rdist = get_accuracy(
                pred_bmasks, gt_bmasks, pred_rmasks, gt_rmasks,
                config.angle_res)
            this_bacc += bacc
            this_bpre += bpre
            this_brec += brec
            this_bf1 += bf1
            this_rdist += rdist

            if batch_idx % 20 == 0:
                print('[{0}| {1:3d}/{2:3d}] train_loss: {3:6.3f} '
                      '(b={4:.3f}, r={5:.3f}) | avg_loss: {6:6.3f} | '
                      'bacc: {7:.3f}, bpre: {8:.3f}, brec: {9:.3f}, '
                      'bf1: {10:.3f}, rdist: {11:.3f}'.format(
                    epoch, batch_idx, total_batches, loss.data,
                    bmloss, rmloss, train_loss / (batch_idx + 1),
                    this_bacc / (batch_idx + 1),
                    this_bpre / (batch_idx + 1),
                    this_brec / (batch_idx + 1),
                    this_bf1 / (batch_idx + 1),
                    this_rdist / (batch_idx + 1)))

        # testing
        print('\nTest')
        spnet.eval()
        test_loss = 0

        this_bacc = 0
        this_bpre = 0
        this_brec = 0
        this_bf1 = 0
        this_rdist = 0

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

            bacc, bpre, brec, bf1, rdist = get_accuracy(
                pred_bmasks, gt_bmasks, pred_rmasks, gt_rmasks,
                config.angle_res)
            this_bacc += bacc
            this_bpre += bpre
            this_brec += brec
            this_bf1 += bf1
            this_rdist += rdist

            if batch_idx % 20 == 0:
                print('[{0}| {1:3d}/{2:3d}]  test_loss: {3:6.3f} '
                      '(b={4:.3f}, r={5:.3f}) | avg_loss: {6:6.3f} | '
                      'bacc: {7:.3f}, bpre: {8:.3f}, brec: {9:.3f}, '
                      'bf1: {10:.3f}, rdist: {11:.3f}'.format(
                    epoch, batch_idx, total_batches, loss.data,
                    bmloss, rmloss, test_loss / (batch_idx + 1),
                    this_bacc / (batch_idx + 1),
                    this_bpre / (batch_idx + 1),
                    this_brec / (batch_idx + 1),
                    this_bf1 / (batch_idx + 1),
                    this_rdist / (batch_idx + 1)))

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

