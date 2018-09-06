#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt
import maxflow


def get_background_mask(img):
    dist_img = img

    g = maxflow.Graph[float]()
    dist_thresh = 0.42

    nodeids = g.add_grid_nodes(dist_img.shape)

    mu = 0.5
    # g.add_grid_edges(nodeids, mu)
    for r in range(0, dist_img.shape[0]):
        for c in range(0, dist_img.shape[0] - 1):
            g.add_edge(c, c + 1, mu, mu)

    fg_weights = np.zeros_like(dist_img)
    bg_weights = np.zeros_like(dist_img)

    # For sure foreground
    fg_weights[dist_img > dist_thresh] = 1.5
    bg_weights[dist_img > dist_thresh] = 1.0

    # Likely background
    fg_weights[dist_img < dist_thresh] = 1.0
    bg_weights[dist_img < dist_thresh] = 1.16

    g.add_grid_tedges(nodeids, fg_weights, bg_weights)

    g.maxflow()
    sgm = g.get_grid_segments(nodeids)
    return ~sgm, dist_img, fg_weights, bg_weights


def cut_image_by_graph_cut(img):
    fg_mask, dist_img, fg_weights, bg_weights = get_background_mask(img)
    # rst = np.zeros_like(img)
    # rst = np.where(fg_mask, img, np.ones(img.shape) * 255)
    return fg_mask


def test():
    print('test_opencv')

    img_path = '../data/skewering_positions/cropped_images/seta_00030_apple_0.jpg'

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fg_mask = cut_image_by_graph_cut(img)
    rst_img = img.copy()
    rst_img[~fg_mask] = 0

    plt.figure(figsize=(12, 10))
    plt.subplot(211)
    plt.imshow(img)
    plt.subplot(212)
    plt.imshow(rst_img)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test()
