#!/usr/bin/env python3

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import os
import shutil
import math
import numpy as np
import argparse

import torch
import torch.optim as optim
import torchvision.transforms as transforms

from bite_selection_package.model.spanet import SPANet, DenseSPANet
from bite_selection_package.model.spanet_dataset import SPANetDataset
from bite_selection_package.model.spanet_loss import SPANetLoss
from bite_selection_package.config import spanet_config as config


def arr2str(vec, format_str='.3f'):
    points = ' '.join(['{{0:{}}}'.format(format_str).format(x) for x in vec[:4]])
    actions = ' '.join(['{{0:{}}}'.format(format_str).format(x) for x in vec[4:]])
    return points + ' | ' + actions


def test_spanet():
    print('test_spanet')
    print('use cuda: {}'.format(config.use_cuda))
    print('use densenet: {}'.format(config.use_densenet))
    print('use rgb: {}'.format(config.use_rgb))
    print('use depth: {}'.format(config.use_depth))
    print('use wall: {}'.format(config.use_wall))

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


    config.excluded_item = "broccoli"

    exp_mode = 'normal'
    if config.excluded_item:
        exp_mode = 'test'

    print('load SPANetDataset')
    trainset = SPANetDataset(
        img_dir=config.img_dir,
        depth_dir=config.depth_dir,
        ann_dir=config.ann_dir,
        success_rate_map_path=config.success_rate_map_path,
        img_res=config.img_res,
        list_filepath=train_list_filepath,
        train=False,
        exp_mode=exp_mode,
        excluded_item=config.excluded_item,
        transform=transform,
        use_rgb=config.use_rgb,
        use_depth=config.use_depth,
        use_wall=config.use_wall)

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
        transform=transform,
        use_rgb=config.use_rgb,
        use_depth=config.use_depth,
        use_wall=config.use_wall)

    if config.use_densenet:
        spanet = DenseSPANet()
    else:
        spanet = SPANet(use_rgb=config.use_rgb, use_depth=config.use_depth, use_wall=config.use_wall)

    checkpoint_path = config.checkpoint_best_filename
    print("Checkpoint Path: " + checkpoint_path)
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

    acc_midpoint_err = list()
    acc_action_best = 0.
    acc_action_spanet = 0.
    acc_action_random = 0.
    acc_action_dist = list()
    acc_rotation_err = list()

    pv_sr_list = list()

    # calculate test accuracies
    def calc_accuracies(pred_vector, gt_vector):
        nonlocal acc_midpoint_err
        nonlocal acc_action_best, acc_action_spanet, acc_action_dist, acc_action_random
        nonlocal acc_rotation_err
        nonlocal pv_sr_list

        pv = pred_vector.cpu().detach()[0]
        pv_p1, pv_p2, pv_sr = pv[:2], pv[2:4], pv[4:]
        gv = gt_vector.cpu().detach()[0]
        gv_p1, gv_p2, gv_sr = gv[:2], gv[2:4], gv[4:]

        pv_sr_list.append(pv_sr.numpy().reshape((1, 6)))

        pv_midpoint = (pv_p1 + pv_p2) * 0.5
        gv_midpoint = (gv_p1 + gv_p2) * 0.5

        this_midpoint_err = ((pv_midpoint - gv_midpoint) ** 2).sum().sqrt().item()
        acc_midpoint_err.append(this_midpoint_err)

        acc_action_best += 1.0
        if gv_sr[pv_sr.argmax()] == gv_sr[gv_sr.argmax()]:
            acc_action_spanet += 1.0
        else:
            pass
            #print("Bad Action Percent: " + str(pv_sr.argmax()))

        acc_action_random += 1.0/6.0

    if config.excluded_item:
    #if False:
        trainset_len = trainset.num_samples
        total_test_samples += trainset_len

        # over training set
        for idx in range(trainset.num_samples):
            rgb, depth, gt_vector, loc_type = trainset[idx]
            rgb = torch.stack([rgb]) if rgb is not None else None
            depth = torch.stack([depth]) if depth is not None else None
            gt_vector = torch.stack([gt_vector])
            loc_type = torch.stack([loc_type])
            if config.use_cuda:
                rgb = rgb.cuda() if rgb is not None else None
                depth = depth.cuda() if depth is not None else None
                gt_vector = gt_vector.cuda()
                loc_type = loc_type.cuda()

            pred_vector, feature_map = spanet(rgb, depth, loc_type)
            loss = criterion(pred_vector, gt_vector)
            test_loss += loss.data

            feature_map = feature_map.cpu().data.numpy()

            calc_accuracies(pred_vector, gt_vector)

            pred_vector_str = arr2str(pred_vector.cpu().data.numpy()[0])
            gt_vector_str = arr2str(gt_vector.cpu().data.numpy()[0])
            if idx % 10 == 0:
                print('[{0:3d}/{1:3d}] loss: {2:6.3f}'.format(
                    idx + 1, total_test_samples, loss.data))
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

            feat_save_path = os.path.join(
                sample_feature_dir, 'sample_{0:04d}.npy'.format(idx))
            np.save(feat_save_path, feature_map)

    # over test set
    for idx in range(testset.num_samples):
        rgb, depth, gt_vector, loc_type = testset[idx]
        rgb = torch.stack([rgb]) if rgb is not None else None
        depth = torch.stack([depth]) if depth is not None else None
        gt_vector = torch.stack([gt_vector])
        loc_type = torch.stack([loc_type])
        if config.use_cuda:
            rgb = rgb.cuda() if rgb is not None else None
            depth = depth.cuda() if depth is not None else None
            gt_vector = gt_vector.cuda()
            loc_type = loc_type.cuda()

        pred_vector, feature_map = spanet(rgb, depth, loc_type)
        loss = criterion(pred_vector, gt_vector)
        test_loss += loss.data

        calc_accuracies(pred_vector, gt_vector)

        pred_vector_str = arr2str(pred_vector.cpu().data.numpy()[0])
        gt_vector_str = arr2str(gt_vector.cpu().data.numpy()[0])
        if idx % 10 == 0:
            print('[{0:3d}/{1:3d}] loss: {2:6.3f}'.format(
                trainset_len + idx + 1, total_test_samples, loss.data))
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
            sample_vector_dir, 'sample_{0:04d}.txt'.format(
                trainset_len + idx))
        with open(vec_save_path, 'w') as f_vec:
            f_vec.write(pred_vector_str)
            f_vec.write('\n')
            f_vec.write(gt_vector_str)
            f_vec.write('\n')
            f_vec.close()

        feat_save_path = os.path.join(
            sample_feature_dir, 'sample_{0:04d}.npy'.format(
                trainset_len + idx))
        np.save(feat_save_path, feature_map.cpu())

    print(checkpoint_path)
    print('Average loss: {0:6.3f}'.format(
        test_loss / total_test_samples))
    print('Success Rate - Best Action: {0:6.3f}'.format(
        acc_action_best / total_test_samples))
    print('Success Rate - SPANet: {0:6.3f}'.format(
        acc_action_spanet / total_test_samples))
    print('Success Rate - Random: {0:6.3f}'.format(
        acc_action_random / total_test_samples))
    print('Accuracy - Midpoint error: {0:6.3f}, (std: {2:6.3f}, max: {1:6.3f})'.format(
        np.mean(acc_midpoint_err),
        np.std(acc_midpoint_err),
        np.max(acc_midpoint_err)))

    pv_sr_list_2d = np.array(pv_sr_list).reshape((len(pv_sr_list), 6))

    mean = np.mean(pv_sr_list_2d, axis=0)
    print('Mean ESR Per Action: ' + str(mean))
    print('STE: ' + str(np.sqrt(np.multiply(mean, 1.0 - mean) / float(total_test_samples))))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-g', '--gpu_id', default=config.gpu_id,
                    help="target gpu index to run this model")
    ap.add_argument('-e', '--exc_id', default=0,
                    type=int, help="idx of an item to exclude")
    args = ap.parse_args()

    if args.gpu_id == '-1':
        config.use_cuda = False
    else:
        config.use_cuda = True
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    config.excluded_item = config.items[args.exc_id]
    #config.set_project_prefix()

    test_spanet()
