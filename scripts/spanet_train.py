#!/usr/bin/env python3

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import os
import shutil
import math
import argparse

import torch
import torch.optim as optim
import torchvision.transforms as transforms

from bite_selection_package.model.spanet import SPANet, DenseSPANet
from bite_selection_package.model.spanet_dataset import SPANetDataset
from bite_selection_package.model.spanet_loss import SPANetLoss
from bite_selection_package.config import spanet_config as config


def train_spanet():
    print('train_spanet')
    print('use cuda: {}'.format(config.use_cuda))
    print('use densenet: {}'.format(config.use_densenet))

    train_list_filepath = config.train_list_filepath
    test_list_filepath = config.test_list_filepath

    if not os.path.exists(train_list_filepath):
        print('list files are missing. generating list files.')
        ann_filenames_all = os.listdir(config.ann_dir)
        ann_filenames = list()
        for filename in ann_filenames_all:
            img_filename = os.path.join(config.img_dir, filename[:-4] + '.png')
            if not os.path.exists(img_filename):
                continue
            ann_filenames.append(filename)

        train_len = int(len(ann_filenames) * 0.9)
        with open(train_list_filepath, 'w') as f_train:
            for idx in range(train_len):
                f_train.write(ann_filenames[idx])
                f_train.write('\n')
            f_train.close()
        with open(test_list_filepath, 'w') as f_test:
            for idx in range(train_len, len(ann_filenames)):
                f_test.write(ann_filenames[idx])
                f_test.write('\n')
            f_test.close()

    checkpoint_path = config.checkpoint_filename
    checkpoint_path_best = config.checkpoint_best_filename

    # sample_dir_name = os.path.join('samples', config.project_prefix)
    # sample_dir = os.path.join(
    #     config.project_dir, sample_dir_name)

    # sample_img_dir = os.path.join(sample_dir, 'cropped_images')
    # sample_ann_dir = os.path.join(sample_dir, 'masks')
    # if os.path.exists(sample_dir):
    #     shutil.rmtree(sample_dir)
    # os.makedirs(sample_dir)
    # os.makedirs(sample_img_dir)
    # os.makedirs(sample_ann_dir)

    transform = transforms.Compose([
        transforms.ToTensor()])
    # transforms.Normalize((0.562, 0.370, 0.271), (0.332, 0.302, 0.281))])

    exp_mode = 'full' if config.excluded_item is None else 'exclude'

    print('load SPANetDataset')
    trainset = SPANetDataset(
        img_dir=config.img_dir,
        depth_dir=config.depth_dir,
        ann_dir=config.ann_dir,
        success_rate_map_path=config.success_rate_map_path,
        img_res=config.img_res,
        list_filepath=train_list_filepath,
        train=True,
        exp_mode=exp_mode,
        excluded_item=config.excluded_item,
        transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.train_batch_size,
        shuffle=True, num_workers=8,
        collate_fn=trainset.collate_fn)

    testset = SPANetDataset(
        img_dir=config.img_dir,
        depth_dir=config.depth_dir,
        ann_dir=config.ann_dir,
        success_rate_map_path=config.success_rate_map_path,
        img_res=config.img_res,
        list_filepath=test_list_filepath,
        train=False,
        exp_mode=exp_mode,
        excluded_item=config.excluded_item,
        transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config.test_batch_size,
        shuffle=True, num_workers=8,
        collate_fn=testset.collate_fn)

    if config.use_densenet:
        spanet = DenseSPANet()
    else:
        spanet = SPANet()

    best_loss = float('inf')
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        print('Resuming from checkpoint \"{}\"'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        spanet.load_state_dict(checkpoint['net'])
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']

    spanet = torch.nn.DataParallel(
        spanet, device_ids=range(torch.cuda.device_count()))
    if config.use_cuda:
        spanet = spanet.cuda()

    criterion = SPANetLoss()
    optimizer = optim.SGD(spanet.parameters(), lr=1e-3,
                          momentum=0.9, weight_decay=1e-4)

    print('training set: {}'.format(trainloader.dataset.num_samples))
    print('test set: {}'.format(testloader.dataset.num_samples))

    for epoch in range(start_epoch, start_epoch + 100):
        # training
        print('\nEpoch: {} | {}'.format(epoch, config.project_prefix))
        spanet.train()
        # spanet.module.freeze_bn()
        train_loss = 0

        total_batches = int(math.ceil(
            trainloader.dataset.num_samples / trainloader.batch_size))

        for batch_idx, batch_items in enumerate(trainloader):
            rgb_imgs = batch_items[0]
            depth_imgs = batch_items[1]
            gt_vectors = batch_items[2]

            if config.use_cuda:
                rgb_imgs = rgb_imgs.cuda() if rgb_imgs is not None else None
                depth_imgs = depth_imgs.cuda() if depth_imgs is not None else None
                gt_vectors = gt_vectors.cuda()

            optimizer.zero_grad()
            pred_vectors, _ = spanet(rgb_imgs, depth_imgs)

            loss = criterion(pred_vectors, gt_vectors)
            loss.backward()
            optimizer.step()

            train_loss += loss.data

            if batch_idx % 20 == 0:
                print('[{0}| {1:3d}/{2:3d}] train_loss: {3:6.3f} '
                      '| avg_loss: {4:6.3f}'.format(
                    epoch, batch_idx, total_batches, loss.data,
                    train_loss / (batch_idx + 1)))

        # testing
        print('\nTest')
        spanet.eval()
        test_loss = 0

        total_batches = int(math.ceil(
            testloader.dataset.num_samples / testloader.batch_size))

        for batch_idx, batch_items in enumerate(testloader):
            rgb_imgs = batch_items[0]
            depth_imgs = batch_items[1]
            gt_vectors = batch_items[2]

            if config.use_cuda:
                rgb_imgs = rgb_imgs.cuda() if rgb_imgs is not None else None
                depth_imgs = depth_imgs.cuda() if depth_imgs is not None else None
                gt_vectors = gt_vectors.cuda()

            optimizer.zero_grad()
            pred_vectors, _ = spanet(rgb_imgs, depth_imgs)

            loss = criterion(pred_vectors, gt_vectors)

            test_loss += loss.data

            if batch_idx % 20 == 0:
                print('[{0}| {1:3d}/{2:3d}] test_loss: {3:6.3f} '
                      '| avg_loss: {4:6.3f}'.format(
                    epoch, batch_idx, total_batches, loss.data,
                    test_loss / (batch_idx + 1)))

        # save checkpoint
        state = {
            'net': spanet.module.state_dict(),
            'loss': test_loss,
            'epoch': epoch, }
        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state, checkpoint_path)

        test_loss /= len(testloader)
        print('Avg test_loss: {0:.6f}'.format(test_loss))
        if test_loss < best_loss:
            print('Saving best checkpoint..')
            state = {
                'net': spanet.module.state_dict(),
                'loss': test_loss,
                'epoch': epoch, }
            if not os.path.exists(os.path.dirname(checkpoint_path_best)):
                os.makedirs(os.path.dirname(checkpoint_path_best))
            torch.save(state, checkpoint_path_best)
            best_loss = test_loss


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-g', '--gpu_id', default=config.gpu_id,
                    help="target gpu index to run this model")
    ap.add_argument('-e', '--exc_id', default=0, type=int,
                    help="idx of an item to exclude")
    args = ap.parse_args()

    if args.gpu_id == '-1':
        config.use_cuda = False
    else:
        config.use_cuda = True
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    config.excluded_item = config.items[args.exc_id]
    config.set_project_prefix()

    train_spanet()
