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
from model.spanet import SPANet, DenseSPANet
from model.spanet_dataset import SPANetDataset
from model.spanet_loss import SPANetLoss
from config import spanet_config as config


os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id


def test_spanet():
    print('test_spanet')
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
    sample_dir = os.path.join(config.project_dir, sample_dir_name)

    sample_image_dir = os.path.join(sample_dir, 'image')
    sample_ann_dir = os.path.join(sample_dir, 'ann')
    sample_feature_dir = os.path.join(sample_dir, 'feature')
    if os.path.exists(sample_dir):
        shutil.rmtree(sample_dir)
    os.makedirs(sample_image_dir)
    os.makedirs(sample_ann_dir)
    os.makedirs(sample_feature_dir)

    transform = transforms.Compose([
        transforms.ToTensor()])
    # transforms.Normalize((0.562, 0.370, 0.271), (0.332, 0.302, 0.281))])

    exp_mode = 'normal'
    if config.excluded_item:
        exp_mode = 'test'

    print('load SPANetDataset')
    trainset = SPANetDataset(
        list_filepath=train_list_filepath,
        train=False,
        exp_mode=exp_mode,
        transform=transform)

    testset = SPANetDataset(
        list_filepath=test_list_filepath,
        train=False,
        exp_mode=exp_mode,
        transform=transform)

    if config.use_densenet:
        spanet = DenseSPANet()
    else:
        spanet = SPANet()

    if os.path.exists(checkpoint_path):
        print('Resuming from checkpoint \"{}\"'.format(checkpoint_path_best))
        checkpoint = torch.load(checkpoint_path_best)
        spanet.load_state_dict(checkpoint['net'])
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']

    # spanet = torch.nn.DataParallel(
    #     spanet, device_ids=range(torch.cuda.device_count()))
    if config.use_cuda:
        spanet = spanet.cuda()

    criterion = SPANetLoss()

    print('training set: {}'.format(trainset.num_samples))
    print('test set: {}'.format(testset.num_samples))

    spanet.eval()

    total_test_samples = trainset.num_samples + testset.num_samples
    test_loss = 0

    # over training set
    for idx in range(trainset.num_samples):
        rgb, depth, gt_vector = trainset[idx]
        rgb = torch.stack([rgb]) if rgb is not None else None
        depth = torch.stack([depth]) if depth is not None else None
        gt_vector = torch.stack([gt_vector])
        if config.use_cuda:
            rgb = rgb.cuda() if rgb is not None else None
            depth = depth.cuda() if depth is not None else None
            gt_vector = gt_vector.cuda()

        pred_vector, feature_map = spanet(rgb, depth)
        loss = criterion(pred_vector, gt_vector)
        test_loss += loss.data

        print('[{0:3d}/{1:3d}] loss: {2:6.3f}'.format(
            idx + 1, total_test_samples, loss.data))
        print(['{0:.3f}'.format(x) for x in pred_vector.cpu().data.numpy()[0]])
        print(['{0:.3f}'.format(x) for x in gt_vector.cpu().data.numpy()[0]])
        print('')

        # save this sample
        sample_img = rgb if rgb is not None else depth
        img_save_path = os.path.join(
            sample_image_dir, 'sample_{0:04d}.png'.format(idx))
        pil_img = transforms.ToPILImage()(sample_img.cpu()[0])
        pil_img.save(img_save_path)
        ann_save_path = os.path.join(
            sample_ann_dir, 'sample_{0:04d}.txt'.format(idx))
        with open(ann_save_path, 'w') as f_ann:
            f_ann.write(' '.join(list(map(
                str, pred_vector.cpu().detach().numpy()[0, :4]))))
            f_ann.write('\n')
            f_ann.close()

    # over test set
    for idx in range(testset.num_samples):
        rgb, depth, gt_vector = testset[idx]
        rgb = torch.stack([rgb]) if rgb is not None else None
        depth = torch.stack([depth]) if depth is not None else None
        gt_vector = torch.stack([gt_vector])
        if config.use_cuda:
            rgb = rgb.cuda() if rgb is not None else None
            depth = depth.cuda() if depth is not None else None
            gt_vector = gt_vector.cuda()

        pred_vector, feature_map = spanet(rgb, depth)
        loss = criterion(pred_vector, gt_vector)
        test_loss += loss.data

        print('[{0:3d}/{1:3d}] loss: {2:6.3f}'.format(
            trainset.num_samples + idx + 1, total_test_samples, loss.data))
        print(['{0:.3f}'.format(x) for x in pred_vector.cpu().data.numpy()[0]])
        print(['{0:.3f}'.format(x) for x in gt_vector.cpu().data.numpy()[0]])
        print('')

        # save this sample
        sample_img = rgb if rgb is not None else depth
        img_save_path = os.path.join(
            sample_image_dir, 'sample_{0:04d}.png'.format(
                trainset.num_samples + idx))
        pil_img = transforms.ToPILImage()(sample_img.cpu()[0])
        pil_img.save(img_save_path)
        ann_save_path = os.path.join(
            sample_ann_dir, 'sample_{0:04d}.txt'.format(
                trainset.num_samples + idx))
        with open(ann_save_path, 'w') as f_ann:
            f_ann.write(' '.join(list(map(
                str, pred_vector.cpu().detach().numpy()[0, :4]))))
            f_ann.write('\n')
            f_ann.close()

    print(checkpoint_path_best)
    print('Average loss: {0:6.3f}'.format(
        test_loss / total_test_samples))


if __name__ == '__main__':
    test_spanet()
