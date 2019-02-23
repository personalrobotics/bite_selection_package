#!/usr/bin/env python3

import os
import numpy as np

from PIL import Image, ImageDraw, ImageFont


sample_dir = '../samples/food_spanet_rgb'
save_dir = os.path.join(sample_dir, 'visualizations')


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
    fnt_title = ImageFont.truetype('Roboto-Regular.ttf', 15)
    fnt = ImageFont.truetype('Roboto-Regular.ttf', 20)
    lr_margin = 5
    tb_margin = 3
    cell_offset = (img_size[0] - lr_margin * 2) / 7.0
    vertical_offset = (margin - tb_margin * 2) / 3.0

    title_top = img_size[1] + tb_margin
    pred_top = img_size[1] + vertical_offset
    gt_top = img_size[1] + vertical_offset * 2

    titles = ['', 'v0', 'v90', 'tv0', 'tv90', 'ta0', 'ta90']
    pred_strs = ['pred'] + list(map(str, pred))
    gt_strs = ['gt'] + list(map(str, gt))
    for idx in range(7):
        col_x = lr_margin + cell_offset * idx
        draw.text(
            (col_x, title_top),
            titles[idx], font=fnt_title, fill=(15, 15, 15, 255))
        if idx > 0:
            draw.rectangle(
                (col_x - 7, pred_top, col_x + cell_offset - 10, pred_top + 23),
                fill=(250, 0, 0, int(float(pred_strs[idx]) * 150)),
                outline=None)
            draw.rectangle(
                (col_x - 7, gt_top, col_x + cell_offset - 10, gt_top + 23),
                fill=(0, 250, 0, int(float(gt_strs[idx]) * 150)),
                outline=None)
        draw.text(
            (col_x, pred_top),
            pred_strs[idx], font=fnt, fill=(0, 0, 0, 255))
        draw.text(
            (col_x, gt_top),
            gt_strs[idx], font=fnt, fill=(0, 0, 0, 255))


def test_draw(img_path, pred_vector, gt_vector):
    img_org = Image.open(img_path)
    if img_org is None:
        return
    new_size = 600
    img_org = img_org.resize((new_size, new_size), resample=Image.BILINEAR)

    img_size = img_org.size

    margin = 80
    img = Image.new('RGBA', (img_size[0], img_size[1] + margin),
                    color=(255, 255, 255))

    img.paste(img_org)

    draw = ImageDraw.Draw(img)

    pred_points = tuple(np.asarray(pred_vector[:4]) * new_size)
    gt_points = tuple(np.asarray(gt_vector[:4]) * new_size)
    draw_vectors(draw, pred_points, gt_points)

    pred_scores = pred_vector[4:]
    gt_scores = gt_vector[4:]
    draw_scores(draw, img_size, margin, pred_scores, gt_scores)

    del draw

    save_path = os.path.join(save_dir, os.path.basename(img_path))
    img.save(save_path)
    print(save_path)


def generate_visualizations():
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image_dir = os.path.join(sample_dir, 'cropped_images')
    vector_dir = os.path.join(sample_dir, 'vector')

    image_names = os.listdir(image_dir)

    for image_name in image_names:
        image_path = os.path.join(image_dir, image_name)
        vector_path = os.path.join(
            vector_dir, image_name[:-4] + '.txt')

        with open(vector_path, 'r') as f_vec:
            lines = list(map(str.strip, f_vec.readlines()))
            f_vec.close()
        if lines is None or len(lines) == 0:
            return

        pred_vector = list(map(float, lines[0].split()))
        gt_vector = list(map(float, lines[1].split()))

        test_draw(image_path, pred_vector, gt_vector)


if __name__ == '__main__':
    generate_visualizations()
