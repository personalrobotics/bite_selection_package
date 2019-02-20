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


from math import atan2, cos, sin, sqrt, pi
def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)


def getOrientation(data_pts, img):
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))


    cv2.circle(img, cntr, 3, (255, 0, 0), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (0, 0, 255), 5)
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians

    return angle


def test():
    print('test_opencv2')

    img_path = '../data/skewering_positions_general/cropped_images/tilted_vertical_skewer_isolated+data_collection+carrot-angle-0-trial-3+color+image_0_carrots_03260122.jpg'

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    imgray = cv2.bilateralFilter(imgray, 11, 17, 17)

    edged = cv2.Canny(imgray, 20, 80)
    edged = np.asarray(edged)

    rows, cols = np.where(edged == 255)

    data_pts = np.asarray(zip(rows, cols), dtype=np.float64)

    # _, contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # pts = contours[0]
    # sz = len(pts)
    # data_pts = np.empty((sz, 2), dtype=np.float64)
    # for i in range(data_pts.shape[0]):
    #     data_pts[i,0] = pts[i,0,0]
    #     data_pts[i,1] = pts[i,0,1]
    # print(data_pts)

    getOrientation(data_pts, img)

    plt.figure(figsize=(12, 10))
    plt.imshow(img)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test()
