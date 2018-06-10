from __future__ import print_function
from __future__ import division

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import sys


class SPSampler(object):
    def __init__(self,
                 base_dir='../data/skewering_positions',
                 bbox_base_dir='../data/bounding_boxes/'):
        self.base_dir = base_dir
        self.bbox_base_dir = bbox_base_dir

        self.bbox_ann_dir = os.path.join(
                self.bbox_base_dir, 'annotations/xmls')
        self.bbox_img_dir = os.path.join(
                self.bbox_base_dir, 'images')

        self.ann_dir = os.path.join(self.base_dir, 'annotations')
        self.img_dir = os.path.join(self.base_dir, 'cropped_images')

        self.cur_img_name = None
        self.is_clicked = False

        samplable_objs = [
            'apple', 'apricot', 'banana', 'bell_pepper', 'blackberry',
            'cantalope', 'carrot', 'celery', 'cherry_tomato', 'egg',
            'grape', 'melon', 'strawberry']
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

    def generate_cropped_images(self):
        print('-- generate_cropped_images ---------------------------------\n')
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

            bboxes = dict()
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            for node in root:
                if node.tag == 'object':
                    obj_name = node.find('name').text
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
                        self.img_dir, '{}_{}_{}.jpg'.format(
                            xml_filename[:-4], obj_name, bidx))
                    print(save_path)
                    cv2.imwrite(save_path, cropped_img)
        print('-- generate_cropped_images finished ------------------------\n')

    def sampling_skewering_positions(self, skip=True):
        print('-- sampling_skewering_positions ----------------------------\n')
        img_filename_list = sorted(os.listdir(self.img_dir))

        self.cur_idx = 0
        while self.cur_idx < len(img_filename_list):
            img_filename = img_filename_list[self.cur_idx]
            print('[{}] {}'.format(self.cur_idx, img_filename))

            if not img_filename.endswith('.jpg'):
                self.cur_idx += 1
                continue
            self.cur_img_name = img_filename[:-4]
            if self.cur_img_name.split('_')[2] not in self.samplable:
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
                saved_values = map(float, content)

            img_filepath = os.path.join(self.img_dir, img_filename)
            self.plot_img = cv2.imread(img_filepath)
            self.plot_img = cv2.cvtColor(self.plot_img, cv2.COLOR_BGR2RGB)

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
            self.cur_fig = plt.figure(0, figsize=[6, 6])
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

        elif event.key == 'd':
            print('Go to the next image')
            self.plot_guide.set_visible(False)
            plt.close(self.cur_fig)
            self.cur_idx += 1

        else:
            print('\"{}\" key pressed.')
            print('Available actions keys:')
            print('  (1) Save : \'y\', \'enter\', \'space\'')
            print('  (2) Reset : \'n\', \'escape\', \'backspace\'')
            print('  (3) Next : \'d\'')
            print('  (4) Prev : \'a\'\n')


options = dict()
options['all'] = 'running both cropping and sampling'
options['crop'] = 'generating cropped images'
options['sample'] = 'sampling skewering positions'
options['view'] = 'checking saved skewering positions'


def print_usage():
    print('Usage:')
    print('    python {} <option>\n'.format(sys.argv[0]))
    print('Available options:')
    print('    {0:11s}{1}'.format('all', options['all']))
    print('    {0:11s}{1}'.format('crop', options['crop']))
    print('    {0:11s}{1}'.format('sample', options['sample']))
    print('    {0:11s}{1}'.format('view', options['view']))


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

    else:
        print_usage()
