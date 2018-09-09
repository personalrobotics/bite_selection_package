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


class SPSampler(object):
    def __init__(self,
                 base_dir='../data/skewering_positions_c8',
                 bbox_base_dir='../data/bounding_boxes_c8/'):
        self.base_dir = base_dir
        self.bbox_base_dir = bbox_base_dir

        project_prefix = 'food_c8'

        self.bbox_ann_dir = os.path.join(
                self.bbox_base_dir, 'annotations/xmls')
        self.bbox_img_dir = os.path.join(
                self.bbox_base_dir, 'images')

        self.ann_dir = os.path.join(self.base_dir, 'annotations')
        self.img_dir = os.path.join(self.base_dir, 'cropped_images')

        self.label_map_path = '../data/{}_label_map.pbtxt'.format(project_prefix)

        self.listdataset = list()
        self.listdataset_train_path = '../data/{}_ann_train.txt'.format(project_prefix)
        self.listdataset_test_path = '../data/{}_ann_test.txt'.format(project_prefix)

        self.cur_img_name = None
        self.is_clicked = False

        if project_prefix.endswith('c8'):
            samplable_objs = [
                'banana', 'cantaloupe', 'carrot', 'celery',
                'egg', 'green_grape', 'strawberry']
        elif project_prefix.endswith('c9'):
            samplable_objs = [
                'apple', 'banana', 'carrot', 'celery',
                'egg', 'grape_purple', 'grape_green', 'melon']
        else:
            samplable_objs = [
                'apple', 'apricot', 'banana', 'bell_pepper', 'blackberry',
                'broccoli', 'cantalope', 'carrot', 'cauliflower', 'celery',
                'cherry_tomato', 'egg', 'grape_purple', 'grape_green',
                'melon', 'strawberry']

        self.samplable = dict()
        for obj_name in samplable_objs:
            self.samplable[obj_name] = True

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
        self.mask_org_dir = os.path.join(self.base_dir, 'masks_renamed')

        xml_filenames = sorted(os.listdir(self.bbox_ann_dir))
        for xidx, xml_filename in enumerate(xml_filenames):
            if not xml_filename.endswith('.xml'):
                continue
            xml_file_path = os.path.join(self.bbox_ann_dir, xml_filename)
            img_file_path = os.path.join(
                self.bbox_img_dir, xml_filename[:-4] + '.jpg')

            img = cv2.imread(img_file_path)

            print('[{}/{}] {}'.format(
                xidx + 1, len(xml_filenames), xml_file_path))

            this_ann_line = xml_filename[:-4] + '.jpg'

            num_boxes = 0
            bboxes = dict()
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            for node in root:
                if node.tag == 'object':
                    obj_name = node.find('name').text
                    if obj_name not in self.samplable:
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
                        self.img_dir, '{0}_{1}_{2:04d}{3:04d}.jpg'.format(
                            xml_filename[:-4], obj_name, xmin, ymin))
                    print(save_path)
                    cv2.imwrite(save_path, cropped_img)

                    mask_path = os.path.join(
                        self.mask_dir, '{0}_{1}_{2:04d}{3:04d}.txt'.format(
                            xml_filename[:-4], obj_name, xmin, ymin))
                    mask_org_path = os.path.join(
                        self.mask_org_dir, '{0}_{1}_{2:04d}{3:04d}.txt'.format(
                            xml_filename[:-4], obj_name, xmin, ymin))
                    # mask_org_path = os.path.join(
                    #     self.mask_org_dir, '{}_{}_{}.txt'.format(
                    #         xml_filename[:-4], obj_name, bidx))
                    if os.path.exists(mask_org_path):
                        shutil.copy(mask_org_path, mask_path)

                    this_ann_line += ' {} {} {} {} {} {}'.format(
                        xmin, ymin, xmax, ymax,
                        label_dict[obj_name], bidx)
                    num_boxes += 1

            if num_boxes > 0:
                self.listdataset.append(this_ann_line)

        random.shuffle(self.listdataset)

        num_trainset = int(len(self.listdataset) * 0.9)
        with open(self.listdataset_train_path, 'w') as f:
            for idx in range(0, num_trainset):
                f.write('{}\n'.format(self.listdataset[idx]))
            f.close()
        with open(self.listdataset_test_path, 'w') as f:
            for idx in range(num_trainset, len(self.listdataset)):
                f.write('{}\n'.format(self.listdataset[idx]))
            f.close()

        print('-- generate_cropped_images finished ------------------------\n')

    def sampling_skewering_positions(self, skip=True):
        print('-- sampling_skewering_positions ----------------------------\n')
        img_filename_list = sorted(os.listdir(self.img_dir))

        self.cur_idx = 0
        while self.cur_idx < len(img_filename_list):
            img_filename = img_filename_list[self.cur_idx]
            print('[{}/{}] {}'.format(
                self.cur_idx, len(img_filename_list), img_filename))

            if not img_filename.endswith('.jpg'):
                self.cur_idx += 1
                continue
            self.cur_img_name = img_filename[:-4]
            if (not self.cur_img_name.startswith('test_') and
                    self.cur_img_name.split('_')[2] not in self.samplable):
                self.cur_idx += 1
                continue

            self.outfile_path = os.path.join(
                self.ann_dir, self.cur_img_name + '.out')

            saved_values = None
            if os.path.exists(self.outfile_path):
                if skip:
                    self.cur_idx += 1
                    continue
                f = open(self.outfile_path, 'r')
                content = f.read().split()
                f.close()

                x = float(content[0])
                y = float(content[1])
                ang = float(content[2])
                saved_values = np.array([x, y, ang])

            img_filepath = os.path.join(self.img_dir, img_filename)
            self.plot_img = cv2.imread(img_filepath)
            self.plot_img = cv2.cvtColor(self.plot_img, cv2.COLOR_BGR2RGB)

            if saved_values is not None and saved_values[0] < 1:
                saved_values[0] *= self.plot_img.shape[0]
                saved_values[1] *= self.plot_img.shape[1]

            self.cur_start_point = [-1, -1]
            self.cur_end_point = [-1, -1]
            self.cur_rotation = -1.0

            self.cur_fig = None
            self.plot_guide = None
            self.reset_plot(saved_values=saved_values)
        print('-- sampling_skewering_positions finished -------------------\n')

    def set_plot_msg(self):
        self.plot_msg.set_text(
            'P: [{0:4.2f}, {1:4.2f}],   R: {2:5.2f}'.format(
                self.cur_start_point[0], self.cur_start_point[1],
                self.cur_rotation))

    def reset_plot(self, saved_values=None):
        if self.cur_fig is None:
            self.cur_fig = plt.figure(1, figsize=[6, 6])
        else:
            self.cur_fig.clf()
        plt.ion()

        self.cur_cid_btn_press = self.cur_fig.canvas.mpl_connect(
            'button_press_event', self.onclick)
        self.cur_cid_btn_release = self.cur_fig.canvas.mpl_connect(
            'button_release_event', self.onrelease)
        self.cur_cid_key_press = self.cur_fig.canvas.mpl_connect(
            'key_press_event', self.onkeypress)

        if saved_values is not None:
            self.cur_start_point = saved_values[:2]
            self.cur_end_point = saved_values[:2]
            self.cur_rotation = saved_values[2]
        else:
            self.cur_start_point = [-1, -1]
            self.cur_end_point = [-1, -1]
            self.cur_rotation = -1.0

        self.plot_guide = plt.text(
            0, self.plot_img.shape[0] * -0.075,
            'Save? [Y]es, [N]o', fontsize=10)
        self.plot_guide.set_visible(False)

        self.plot_msg = plt.text(
            0, self.plot_img.shape[0] * -0.02,
            'P: [{0:4.2f}, {1:4.2f}],   R: {2:5.2f}'.format(
                self.cur_start_point[0], self.cur_start_point[1],
                self.cur_rotation),
            fontsize=13)

        self.plot_sp = plt.plot(
            self.cur_start_point[0], self.cur_start_point[1],
            color='cyan', marker='+', markersize=20, markeredgewidth=3)[0]

        rho = min(self.plot_img.shape[:2]) / 4.0
        theta = np.radians(self.cur_rotation)
        pdiff = [rho * np.sin(theta),
                 rho * np.cos(theta)]
        p0 = [self.cur_start_point[0] - pdiff[0],
              self.cur_start_point[0] + pdiff[0]]
        p1 = [self.cur_start_point[1] - pdiff[1],
              self.cur_start_point[1] + pdiff[1]]
        self.plot_line = plt.plot(
            p0, p1, color='yellow', lineStyle='--', marker='')[0]

        plt.imshow(self.plot_img)
        plt.show(self.cur_fig)

    def disconnect_events(self, cid_list):
        for cid in cid_list:
            self.cur_fig.canvas.mpl_disconnect(cid)

    def onclick(self, event):
        if (event.button == 1 and
                event.xdata is not None and event.ydata is not None):
            self.cur_cid_btn_move = self.cur_fig.canvas.mpl_connect(
                'motion_notify_event', self.onmove)

            self.cur_start_point = [float(event.xdata), float(event.ydata)]
            self.cur_end_point = [float(event.xdata), float(event.ydata)]

            # self.plot_sp = plt.plot(
            #     self.cur_start_point[0], self.cur_start_point[1],
            #     color='cyan', marker='+', markersize=20, markeredgewidth=3)
            self.plot_sp.set_xdata(self.cur_start_point[0])
            self.plot_sp.set_ydata(self.cur_start_point[1])

            self.plot_line.set_visible(False)

            self.cur_rotation = 0.0
            self.set_plot_msg()
        else:
            self.cur_cid_btn_move = None

    def onrelease(self, event):
        if event.button == 1 and self.cur_cid_btn_move is not None:
            if event.xdata is not None and event.ydata is not None:
                self.cur_end_point = [float(event.xdata), float(event.ydata)]

            pdiff = [self.cur_start_point[0] - self.cur_end_point[0],
                     self.cur_start_point[1] - self.cur_end_point[1]]

            self.cur_rotation = np.degrees(
                np.arctan2(pdiff[0], pdiff[1])) % 180
            self.set_plot_msg()

            self.disconnect_events([
                self.cur_cid_btn_press,
                self.cur_cid_btn_move,
                self.cur_cid_btn_release])

            self.plot_guide.set_visible(True)

    def onmove(self, event):
        if event.button == 1:
            if event.xdata is not None and event.ydata is not None:
                self.cur_end_point = [float(event.xdata), float(event.ydata)]

            if not self.plot_line.get_visible():
                self.plot_line.set_visible(True)

            pdiff = [self.cur_start_point[0] - self.cur_end_point[0],
                     self.cur_start_point[1] - self.cur_end_point[1]]

            p0 = [self.cur_start_point[0] - pdiff[0],
                  self.cur_start_point[0] + pdiff[0]]
            p1 = [self.cur_start_point[1] - pdiff[1],
                  self.cur_start_point[1] + pdiff[1]]

            self.plot_line.set_xdata(p0)
            self.plot_line.set_ydata(p1)

            self.cur_rotation = np.degrees(
                np.arctan2(pdiff[0], pdiff[1])) % 180
            self.set_plot_msg()

    def onkeypress(self, event):
        if (event.key == 'y' or event.key == 'Y' or event.key == 'enter' or
                event.key == 'space'):
            self.plot_guide.set_visible(False)
            # self.disconnect_events([self.cur_cid_key_press])

            outfile = open(self.outfile_path, 'w')
            content = '{0:.2f} {1:.2f} {2:.2f}\n'.format(
                    self.cur_start_point[0],
                    self.cur_start_point[1],
                    self.cur_rotation)
            outfile.write(content)
            outfile.close()
            print('New data: [{0}, {1}], {2:.2f}'.format(
                self.cur_start_point[0], self.cur_start_point[1],
                self.cur_rotation))
            print('Saved in: {}'.format(self.outfile_path))
            plt.close(self.cur_fig)
            self.cur_idx += 1

        elif (event.key == 'n' or event.key == 'N' or
                event.key == 'escape' or event.key == 'backspace'):
            print('Reset! Please set the position and rotation again.')
            self.plot_guide.set_visible(False)
            # self.disconnect_events([self.cur_cid_key_press])
            self.reset_plot()

        elif event.key == 'a':
            print('Go to the previous image')
            self.plot_guide.set_visible(False)
            self.cur_idx -= 1
            if self.cur_idx < 0:
                print('This is the first image')
                self.cur_idx = 0
            else:
                plt.close(self.cur_fig)

        elif event.key == 'A':
            print('Go to the previous image (+100)')
            self.plot_guide.set_visible(False)
            self.cur_idx -= 100
            if self.cur_idx < 0:
                self.cur_idx = 0
            plt.close(self.cur_fig)

        elif event.key == 'd':
            print('Go to the next image')
            self.plot_guide.set_visible(False)
            plt.close(self.cur_fig)
            self.cur_idx += 1

        elif event.key == 'D':
            print('Go to the next image (-100)')
            self.plot_guide.set_visible(False)
            plt.close(self.cur_fig)
            self.cur_idx += 100

        elif event.key == 'shift':
            # skip
            self.cur_idx = self.cur_idx

        else:
            print('\"{}\" key pressed.'.format(event.key))
            print('Available actions keys:')
            print('  (1) Save  : \'y\', \'enter\', \'space\'')
            print('  (2) Reset : \'n\', \'escape\', \'backspace\'')
            print('  (3) Next  : \'d\', \'right\'')
            print('  (4) Prev  : \'a\', \'left\'\n')


options = dict()
options['all'] = 'running both cropping and sampling'
options['crop'] = 'generating cropped images'
options['sample'] = 'sampling skewering positions'
options['view'] = 'checking saved skewering positions'
options['view_w_dir'] = 'checking saved skewering positions in the given path'


def print_usage():
    print('Usage:')
    print('    python {} <option>\n'.format(sys.argv[0]))
    print('Available options:')
    print('    {0:18s}{1}'.format('all', options['all']))
    print('    {0:18s}{1}'.format('crop', options['crop']))
    print('    {0:18s}{1}'.format('sample', options['sample']))
    print('    {0:18s}{1}'.format('view', options['view']))
    print('    {0:18s}{1}'.format('view <base_dir>', options['view_w_dir']))


if __name__ == '__main__':
    if (len(sys.argv) == 2 and
            (sys.argv[1] in options)):

        spsampler = SPSampler()

        if sys.argv[1] == 'all' or sys.argv[1] == 'crop':
            spsampler.generate_cropped_images()
        if sys.argv[1] == 'all' or sys.argv[1] == 'sample':
            spsampler.sampling_skewering_positions()
        if sys.argv[1] == 'view':
            spsampler.sampling_skewering_positions(skip=False)

    elif (len(sys.argv) == 3 and
            (sys.argv[1] == 'view') and
            (os.path.exists(sys.argv[2]))):
        spsampler = SPSampler(base_dir=sys.argv[2])
        spsampler.sampling_skewering_positions(skip=False)

    else:
        print_usage()
