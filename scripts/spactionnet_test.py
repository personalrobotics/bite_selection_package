#!/usr/bin/env python3

from __future__ import division
from __future__ import print_function

import sys
import os
import shutil
import math

import torch
import torch.optim as optim
import torchvision.transforms as transforms

sys.path.append(os.path.split(os.getcwd())[0])
from model.spactionnet import SPActionNet, DenseSPActionNet
from model.spactionnet_dataset import SPActionNetDataset
from model.spactionnet_loss import SPActionNetLoss
from config import spactionnet_config as config


os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id


def test_spactionnet():
    print('test_spactionnet')
    print('use cuda: {}'.format(config.use_cuda))
    print('use densenet: {}'.format(config.use_densenet))
    print('use rgb: {}'.format(config.use_rgb))
    print('use depth: {}'.format(config.use_depth))

    train_list_filepath = config.train_list_filepath
    test_list_filepath = config.test_list_filepath

    if not os.path.exists(train_list_filepath):
        print('cannot find {}'.format(train_list_filepath))
        return

    checkpoint_path = config.checkpoint_filename
    checkpoint_path_best = config.checkpoint_best_filename

    sample_dir_name = os.path.join('samples', config.project_prefix)
    sample_dir = os.path.join(
        config.project_dir, sample_dir_name)

    sample_image_dir = os.path.join(sample_dir, 'image')
    sample_feature_dir = os.path.join(sample_dir, 'feature')
    if os.path.exists(sample_dir):
        shutil.rmtree(sample_dir)
    os.makedirs(sample_image_dir)
    os.makedirs(sample_feature_dir)

    transform = transforms.Compose([
        transforms.ToTensor()])
    # transforms.Normalize((0.562, 0.370, 0.271), (0.332, 0.302, 0.281))])

    print('load SPActionNetDataset')
    trainset = SPActionNetDataset(
        list_filepath=train_list_filepath,
        train=True,
        transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.train_batch_size,
        shuffle=True, num_workers=8,
        collate_fn=trainset.collate_fn)

    testset = SPActionNetDataset(
        list_filepath=test_list_filepath,
        train=False,
        transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config.test_batch_size,
        shuffle=True, num_workers=8,
        collate_fn=testset.collate_fn)

    if config.use_densenet:
        spactionnet = DenseSPActionNet()
    else:
        spactionnet = SPActionNet()

    if os.path.exists(checkpoint_path):
        print('Resuming from checkpoint \"{}\"'.format(checkpoint_path_best))
        checkpoint = torch.load(checkpoint_path_best)
        spactionnet.load_state_dict(checkpoint['net'])
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']

    spactionnet = torch.nn.DataParallel(
        spactionnet, device_ids=range(torch.cuda.device_count()))
    if config.use_cuda:
        spactionnet = spactionnet.cuda()

    criterion = SPActionNetLoss()

    print('training set: {}'.format(trainloader.dataset.num_samples))
    print('test set: {}'.format(testloader.dataset.num_samples))

    import IPython; IPython.embed()
    return

    # training
    print('\nEpoch: {} | {}'.format(epoch, config.project_prefix))
    spactionnet.train()
    # spactionnet.module.freeze_bn()
    train_loss = 0

    total_batches = int(math.ceil(
        trainloader.dataset.num_samples / trainloader.batch_size))

    for batch_idx, batch_items in enumerate(trainloader):
        imgs = batch_items[0]
        gt_vectors = batch_items[1]

        if config.use_cuda:
            imgs = imgs.cuda()
            gt_vectors = gt_vectors.cuda()

        optimizer.zero_grad()
        pred_vectors = spactionnet(imgs)

        loss = criterion(pred_vectors, gt_vectors)

        train_loss += loss.data

        if batch_idx % 20 == 0:
            print('[{0}| {1:3d}/{2:3d}] train_loss: {3:6.3f} '
                  '| avg_loss: {4:6.3f}'.format(
                epoch, batch_idx, total_batches, loss.data,
                train_loss / (batch_idx + 1)))

    # testing
    print('\nTest')
    spactionnet.eval()
    test_loss = 0

    total_batches = int(math.ceil(
        testloader.dataset.num_samples / testloader.batch_size))

    for batch_idx, batch_items in enumerate(testloader):
        imgs = batch_items[0]
        gt_vectors = batch_items[1]

        if config.use_cuda:
            imgs = imgs.cuda()
            gt_vectors = gt_vectors.cuda()

        optimizer.zero_grad()
        pred_vectors = spactionnet(imgs)

        loss = criterion(pred_vectors, gt_vectors)

        test_loss += loss.data

        if batch_idx % 20 == 0:
            print('[{0}| {1:3d}/{2:3d}] test_loss: {3:6.3f} '
                  '| avg_loss: {4:6.3f}'.format(
                epoch, batch_idx, total_batches, loss.data,
                test_loss / (batch_idx + 1)))


if __name__ == '__main__':
    test_spactionnet()
