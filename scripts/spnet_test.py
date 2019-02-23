#!/usr/bin/env python3

from __future__ import division
from __future__ import print_function

import numpy as np
import sys
import os
import shutil
import math
import time

import torch
import torchvision.transforms as transforms

from bite_selection_package.model.spnet import SPNet, DenseSPNet
from bite_selection_package.model.spnet_dataset import SPDataset
from bite_selection_package.model.spnet_loss import SPNetLoss
from bite_selection_package.config import spnet_config as config
from bite_selection_package.utils.utils import get_accuracy


os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id


def test_spnet(use_cuda=config.use_cuda):
    print('test_spnet')
    print('use cuda: {}'.format(use_cuda))
    print('use rotation: {}'.format(config.use_rotation))
    print('use densenet: {}'.format(config.use_densenet))

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

    transform = transforms.Compose([
        transforms.ToTensor()])
        # transforms.Normalize((0.562, 0.370, 0.271), (0.332, 0.302, 0.281))])

    exp_mode = 'test' if config.excluded_item else 'normal'

    print('load SPDataset')
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

    total_bacc = list()
    total_bpre = list()
    total_brec = list()
    total_bf1 = list()
    total_rdist = list()

    total_batches = int(math.ceil(
        testloader.dataset.num_samples / testloader.batch_size))

    max_epoch = 10
    start_time = time.time()

    for ei in range(max_epoch):
        test_loss = 0

        this_bacc = 0
        this_bpre = 0
        this_brec = 0
        this_bf1 = 0
        this_rdist = 0

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
                pred_bmasks, gt_bmasks, pred_rmasks, gt_rmasks)
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
                    ei, batch_idx, total_batches, loss.data,
                    bmloss, rmloss, test_loss / (batch_idx + 1),
                    this_bacc / (batch_idx + 1),
                    this_bpre / (batch_idx + 1),
                    this_brec / (batch_idx + 1),
                    this_bf1 / (batch_idx + 1),
                    this_rdist / (batch_idx + 1)))

            total_bacc.append(bacc)
            total_bpre.append(bpre)
            total_brec.append(brec)
            total_bf1.append(bf1)
            if rdist > -1:
                total_rdist.append(rdist)

    elapsed_time = time.time() - start_time
    time_per_image = elapsed_time / testloader.dataset.num_samples / max_epoch

    print('\nFinal | {}'.format(config.project_prefix))

    model_parameters = filter(lambda p: p.requires_grad, spnet.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total parameters: {}'.format(params))

    print('Results: avg_bacc: {0:.3f}, avg_bpre: {1:.3f}, avg_rec: {2:.3f}, '
          'avg_bf1: {3:.3f}, avg_rdist: {4:.3f}\n'.format(
              np.mean(total_bacc),
              np.mean(total_bpre),
              np.mean(total_brec),
              np.mean(total_bf1),
              np.mean(total_rdist)))

    print('time_per_image: {0:.4f}'.format(time_per_image))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        test_spnet(sys.argv[1] != 'nocuda')
    else:
        test_spnet()
