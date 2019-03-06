#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from PIL import Image, ImageDraw, ImageFont


def draw_point(draw, point, psize, fill, width):
    draw.line(
        [point[0] - psize, point[1], point[0] + psize, point[1]],
        fill=fill, width=width)
    draw.line(
        [point[0], point[1] - psize, point[0], point[1] + psize],
        fill=fill, width=width)


def draw_axis(draw, p1, p2, fill=(0, 0, 200, 150), width=4,
              psize=5, pfill=(255, 0, 0, 200), pwidth=3):
    draw.line(p1 + p2, fill=fill, width=width)
    draw_point(draw, p1, psize=psize, fill=pfill, width=pwidth)
    draw_point(draw, p2, psize=psize, fill=pfill, width=pwidth)
    cp = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
    draw_point(draw, cp, psize=psize + 3, fill=pfill, width=pwidth)


def draw_vectors(draw, pred, gt):
    draw_axis(
        draw, pred[:2], pred[2:],
        fill=(200, 0, 0, 150), pfill=(255, 0, 0, 200))
    draw_axis(
        draw, gt[:2], gt[2:],
        fill=(0, 200, 0, 150), pfill=(0, 255, 0, 200))


def draw_scores(draw, img_size, margin, pred, gt):
    fnt_title = ImageFont.truetype('../resource/Roboto-Regular.ttf', 15)
    fnt = ImageFont.truetype('../resource/Roboto-Regular.ttf', 20)
    lr_margin = 5
    tb_margin = 3
    cell_offset = (img_size[0] - lr_margin * 2) / 7.0
    vertical_offset = (margin - tb_margin * 2) / 3.0

    title_top = img_size[1] + tb_margin
    pred_top = img_size[1] + vertical_offset
    gt_top = img_size[1] + vertical_offset * 2

    pred_argmax = np.argmax(pred)
    gt_argmax = np.argmax(gt)

    titles = ['', 'v0', 'v90', 'tv0', 'tv90', 'ta0', 'ta90']
    pred_strs = ['pred'] + list(map(str, pred))
    gt_strs = ['gt'] + list(map(str, gt))
    for idx in range(7):
        col_x = lr_margin + cell_offset * idx
        draw.text(
            (col_x, title_top),
            titles[idx], font=fnt_title, fill=(15, 15, 15, 255))
        if idx > 0:
            pred_outline = None if idx - 1 != pred_argmax else (0, 0, 0, 255)
            draw.rectangle(
                (col_x - 7, pred_top, col_x + cell_offset - 10, pred_top + 23),
                fill=(250, 0, 0, int(float(pred_strs[idx]) * 150)),
                outline=pred_outline)
            gt_outline = None if idx - 1 != gt_argmax else (0, 0, 0, 255)
            draw.rectangle(
                (col_x - 7, gt_top, col_x + cell_offset - 10, gt_top + 23),
                fill=(0, 250, 0, int(float(gt_strs[idx]) * 150)),
                outline=gt_outline)
        draw.text(
            (col_x, pred_top),
            pred_strs[idx], font=fnt, fill=(0, 0, 0, 255))
        draw.text(
            (col_x, gt_top),
            gt_strs[idx], font=fnt, fill=(0, 0, 0, 255))


def test_draw(img_org, save_path, feature_path, pred_vector, gt_vector):
    new_size = 500
    img_org = img_org.resize((new_size, new_size), resample=Image.BILINEAR)

    img_size = img_org.size

    margin = 80
    img = Image.new('RGBA', (img_size[0] * 2, img_size[1] + margin),
                    color=(255, 255, 255))

    img.paste(img_org)

    draw = ImageDraw.Draw(img)

    pred_points = tuple(np.asarray(pred_vector[:4]) * new_size)
    gt_points = tuple(np.asarray(gt_vector[:4]) * new_size)
    draw_vectors(draw, pred_points, gt_points)

    pred_scores = pred_vector[4:]
    gt_scores = gt_vector[4:]
    draw_scores(draw, (new_size * 2, new_size), margin,
                pred_scores, gt_scores)

    feature = np.load(feature_path)

    ft_arr = np.mean(abs(feature[0]), axis=0)
    ft_arr -= np.min(ft_arr)
    ft_arr /= np.max(ft_arr)

    cm_hot = mpl.cm.get_cmap('viridis_r')
    ft_img = cm_hot(ft_arr)
    ft_img = np.uint8(ft_img * 255)

    ft_img = Image.fromarray(ft_img)
    ft_img = ft_img.resize((33, 33), resample=Image.BILINEAR)
    ft_img = ft_img.resize((new_size, new_size))

    ft_img.putalpha(255)
    img_org.putalpha(80)
    ft_img = Image.alpha_composite(ft_img, img_org)
    # ft_img.show()

    img.paste(ft_img, (new_size, 0))

    del draw

    img.save(save_path)
    print(save_path)


def generate_visualizations():
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image_dir = os.path.join(sample_dir, 'cropped_images')
    vector_dir = os.path.join(sample_dir, 'vector')
    feature_dir = os.path.join(sample_dir, 'feature')
    assert os.path.isdir(image_dir), 'cannot find source image directory'
    assert os.path.isdir(vector_dir), 'cannot find source vector directory'
    assert os.path.isdir(feature_dir), 'cannot find source feature directory'

    image_names = os.listdir(image_dir)
    num_images = len(image_names)

    acc_midpoint_err = list()
    acc_action_top1 = 0
    acc_action_valid = 0
    acc_action_dist = list()
    acc_rotation_err = list()

    ## calculate test accuracies ##############################################
    def calc_accuracies(pred_vector, gt_vector, img_size):
        nonlocal acc_midpoint_err
        nonlocal acc_action_top1, acc_action_valid, acc_action_dist
        nonlocal acc_rotation_err

        pv = pred_vector
        pv_p1, pv_p2, pv_sr = pv[:2], pv[2:4], pv[4:]
        gv = gt_vector
        gv_p1, gv_p2, gv_sr = gv[:2], gv[2:4], gv[4:]

        pv_mp = (pv_p1 + pv_p2) * 0.5 * img_size
        gv_mp = (gv_p1 + gv_p2) * 0.5 * img_size

        pv_diff = abs(pv_p1 - pv_p2)
        gv_diff = abs(gv_p1 - gv_p2)

        pv_rot = np.arctan2(pv_diff[1], pv_diff[0])
        gv_rot = np.arctan2(gv_diff[1], gv_diff[0])

        rot_err = abs(pv_rot - gv_rot)
        acc_rotation_err.append(np.degrees(rot_err))

        this_midpoint_err = np.sqrt(np.sum((pv_mp - gv_mp) ** 2))
        acc_midpoint_err.append(this_midpoint_err)

        if np.argmax(pv_sr) == np.argmax(gv_sr):
            acc_action_top1 += 1
        if np.max(pv_sr) > max(np.max(gv_sr) - 0.2, 0.4):
            acc_action_valid += 1
    ###########################################################################

    for image_name in image_names:
        image_path = os.path.join(image_dir, image_name)
        vector_path = os.path.join(
            vector_dir, image_name[:-4] + '.txt')
        feature_path = os.path.join(
            feature_dir, image_name[:-4] + '.npy')

        img_org = Image.open(image_path)
        if img_org is None:
            return

        with open(vector_path, 'r') as f_vec:
            lines = list(map(str.strip, f_vec.readlines()))
            f_vec.close()
        if lines is None or len(lines) == 0:
            return

        pred_vector = np.asarray(
            list(map(float, lines[0].replace('|', '').split())))
        gt_vector = np.asarray(
            list(map(float, lines[1].replace('|', '').split())))

        calc_accuracies(pred_vector, gt_vector, img_org.size)

        save_path = os.path.join(save_dir, os.path.basename(image_path))
        test_draw(img_org, save_path, feature_path, pred_vector, gt_vector)

    print(sample_dir)
    print('Accuracy - Action Top 1: {0:6.3f}'.format(
        acc_action_top1 / num_images))
    print('Accuracy - Action valid: {0:6.3f}'.format(
        acc_action_valid / num_images))
    print('Accuracy - Midpoint error: {0:6.2f} px, (std: {1:6.2f}, max: {2:6.2f})'.format(
        np.mean(acc_midpoint_err),
        np.std(acc_midpoint_err),
        np.max(acc_midpoint_err)))
    print('Accuracy - Midpoint error: {0:6.2f} cm, (std: {1:6.2f}, max: {2:6.2f})'.format(
        np.mean(acc_midpoint_err) / 144. * 5.,
        np.std(acc_midpoint_err) / 144. * 5.,
        np.max(acc_midpoint_err) / 144. * 5.))
    print('Accuracy - Rotation error: {0:6.2f} degrees, (std: {1:6.2f}, max: {2:6.2f}'.format(
        np.mean(acc_rotation_err),
        np.std(acc_rotation_err),
        np.max(acc_rotation_err)))


if __name__ == '__main__':
    sample_dir = '../samples/food_spanet_rgb'
    if len(sys.argv) == 2:
        sample_dir = sys.argv[1]
    sample_dir = os.path.expanduser(sample_dir)
    assert os.path.exists(sample_dir), 'cannot find the sample directory'

    save_dir = os.path.join(sample_dir, 'visualizations')

    generate_visualizations()
    print('finished')
