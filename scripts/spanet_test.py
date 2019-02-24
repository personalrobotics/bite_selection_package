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

from bite_selection_package.model.spanet import SPANet, DenseSPANet
from bite_selection_package.model.spanet_dataset import SPANetDataset
from bite_selection_package.model.spanet_loss import SPANetLoss
from bite_selection_package.config import spanet_config as config


os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id


def arr2str(vec, format_str='.3f'):
    return ' '.join(['{{0:{}}}'.format(format_str).format(x) for x in vec])


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

    sample_dir_name = os.path.join('samples', config.project_prefix)
    sample_dir = os.path.join(config.project_dir, sample_dir_name)

    sample_image_dir = os.path.join(sample_dir, 'cropped_images')
    sample_ann_dir = os.path.join(sample_dir, 'annotations')
    sample_feature_dir = os.path.join(sample_dir, 'feature')
    sample_vector_dir = os.path.join(sample_dir, 'vector')
    if os.path.exists(sample_dir):
        shutil.rmtree(sample_dir)
    os.makedirs(sample_image_dir)
    os.makedirs(sample_ann_dir)
    os.makedirs(sample_feature_dir)
    os.makedirs(sample_vector_dir)

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

    checkpoint_path = config.checkpoint_best_filename
    if os.path.exists(checkpoint_path):
        print('Resuming from checkpoint \"{}\"'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
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

    test_loss = 0

    total_test_samples = testset.num_samples
    trainset_len = 0

    if config.excluded_item:
        trainset_len = trainset.num_samples
        total_test_samples += trainset_len

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
            pred_vector_str = arr2str(pred_vector.cpu().data.numpy()[0])
            gt_vector_str = arr2str(gt_vector.cpu().data.numpy()[0])
            print(pred_vector_str)
            print(gt_vector_str)
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

            vec_save_path = os.path.join(
                sample_vector_dir, 'sample_{0:04d}.txt'.format(idx))
            with open(vec_save_path, 'w') as f_vec:
                f_vec.write(pred_vector_str)
                f_vec.write('\n')
                f_vec.write(gt_vector_str)
                f_vec.write('\n')
                f_vec.close()

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
            trainset_len + idx + 1, total_test_samples, loss.data))

        pred_vector_str = arr2str(pred_vector.cpu().data.numpy()[0])
        gt_vector_str = arr2str(gt_vector.cpu().data.numpy()[0])
        print(pred_vector_str)
        print(gt_vector_str)
        print('')

        # save this sample
        sample_img = rgb if rgb is not None else depth
        img_save_path = os.path.join(
            sample_image_dir, 'sample_{0:04d}.png'.format(
                trainset_len + idx))
        pil_img = transforms.ToPILImage()(sample_img.cpu()[0])
        pil_img.save(img_save_path)

        ann_save_path = os.path.join(
            sample_ann_dir, 'sample_{0:04d}.txt'.format(
                trainset_len + idx))
        with open(ann_save_path, 'w') as f_ann:
            f_ann.write(' '.join(list(map(
                str, pred_vector.cpu().detach().numpy()[0, :4]))))
            f_ann.write('\n')
            f_ann.close()

        vec_save_path = os.path.join(
            sample_vector_dir, 'sample_{0:04d}.txt'.format(idx))
        with open(vec_save_path, 'w') as f_vec:
            f_vec.write(pred_vector_str)
            f_vec.write('\n')
            f_vec.write(gt_vector_str)
            f_vec.write('\n')
            f_vec.close()

    print(checkpoint_path)
    print('Average loss: {0:6.3f}'.format(
        test_loss / total_test_samples))


if __name__ == '__main__':
    test_spanet()
