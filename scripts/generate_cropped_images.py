#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import sys
import random
import shutil
import collections


class SPSampler(object):
    def __init__(self, keyword='spnet_all'):

        self.base_dir = '../data/skewering_positions_{}'.format(keyword)
        self.bbox_base_dir = '../data/bounding_boxes_{}'.format(keyword)

        self.bbox_ann_dir = os.path.join(
                self.bbox_base_dir, 'annotations/xmls')
        self.bbox_img_dir = os.path.join(
                self.bbox_base_dir, 'images')
        self.bbox_depth_dir = os.path.join(
                self.bbox_base_dir, 'depth')

        self.ann_dir = os.path.join(self.base_dir, 'annotations')
        self.img_dir = os.path.join(self.base_dir, 'cropped_images')
        self.depth_dir = os.path.join(self.base_dir, 'cropped_depth')

        self.label_map_path = os.path.join(
            bbox_base_dir, 'food_{}_label_map.pbtxt'.format(keyword))

        self.cur_img_name = None
        self.is_clicked = False

        self.check_dirs()

    def check_dirs(self):
        assert os.path.exists(self.bbox_ann_dir), \
            'cannot find {}'.format(self.bbox_ann_dir)
        assert os.path.exists(self.bbox_img_dir), \
            'cannot find {}'.format(self.bbox_img_dir)
        if not os.path.exists(self.ann_dir):
            os.makedirs(self.ann_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        if not os.path.exists(self.depth_dir):
            os.makedirs(self.depth_dir)

    def load_label_map(self):
        with open(self.label_map_path, 'r') as f:
            content = f.read().splitlines()
            f.close()
        assert content is not None, 'cannot find label map'

        temp = list()
        for line in content:
            line = line.strip()
            if (len(line) > 2 and
                    (line.startswith('id') or
                     line.startswith('name'))):
                temp.append(line.split(':')[1].strip())

        label_dict = dict()
        for idx in range(0, len(temp), 2):
            item_id = int(temp[idx])
            item_name = temp[idx + 1][1:-1]
            label_dict[item_name] = item_id
        return label_dict

    def generate_cropped_images(self):
        print('-- generate_cropped_images ---------------------------------\n')

        label_dict = self.load_label_map()

        self.mask_dir = os.path.join(self.base_dir, 'masks')
        self.mask_org_dir = os.path.join(self.base_dir, 'masks_org')

        xml_filenames = sorted(os.listdir(self.bbox_ann_dir))
        for xidx, xml_filename in enumerate(xml_filenames):
            if not xml_filename.endswith('.xml'):
                continue
            # if xml_filename.startswith('sample'):
            #     continue
            xml_file_path = os.path.join(self.bbox_ann_dir, xml_filename)
            img_file_path = os.path.join(
                self.bbox_img_dir, xml_filename[:-4] + '.png')
            depth_file_path = os.path.join(
                self.bbox_depth_dir, xml_filename[:-4] + '.png')

            img = cv2.imread(img_file_path)
            if img is None:
                continue
            depth = cv2.imread(depth_file_path, flags=cv2.CV_16UC1)
            # if depth is None:
            #     continue

            print('[{}/{}] {}'.format(
                xidx + 1, len(xml_filenames), xml_file_path))

            num_boxes = 0
            bboxes = collections.defaultdict(list)
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            for node in root:
                if node.tag == 'object':
                    obj_name = node.find('name').text
                    obj_name = self.interm_map[obj_name]
                    if obj_name not in label_dict:
                        continue
                    if node.find('bndbox') is None:
                        continue
                    xmin = int(node.find('bndbox').find('xmin').text)
                    ymin = int(node.find('bndbox').find('ymin').text)
                    xmax = int(node.find('bndbox').find('xmax').text)
                    ymax = int(node.find('bndbox').find('ymax').text)
                    if obj_name not in bboxes:
                        bboxes[obj_name] = list()
                    bboxes[obj_name].append([xmin, ymin, xmax, ymax])

            margin = 2
            for obj_name in sorted(bboxes):
                bbox_list = bboxes[obj_name]
                print(obj_name, bbox_list)
                for bidx, bbox in enumerate(bbox_list):
                    xmin, ymin, xmax, ymax = bbox
                    xmin = max(0, xmin - margin)
                    ymin = max(0, ymin - margin)
                    xmax = min(img.shape[1], xmax + margin)
                    ymax = min(img.shape[0], ymax + margin)

                    cropped_img = img[ymin:ymax, xmin:xmax]
                    save_path = os.path.join(
                        self.img_dir, '{0}_{1}_{2:04d}{3:04d}.png'.format(
                            xml_filename[:-4], obj_name, xmin, ymin))
                    print(save_path)
                    cv2.imwrite(save_path, cropped_img)

                    if depth is not None:
                        cropped_depth = depth[ymin:ymax, xmin:xmax]
                        save_depth_path = os.path.join(
                            self.depth_dir, '{0}_{1}_{2:04d}{3:04d}.png'.format(
                                xml_filename[:-4], obj_name, xmin, ymin))
                        print(save_depth_path)
                        cv2.imwrite(save_depth_path, cropped_depth)

        print('-- generate_cropped_images finished ------------------------\n')


def print_usage():
    print('Usage:')
    print('    python {} <keyword>\n'.format(sys.argv[0]))


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1]:
        spsampler = SPSampler(sys.argv[1])
        spsampler.generate_cropped_images()

    else:
        print_usage()
