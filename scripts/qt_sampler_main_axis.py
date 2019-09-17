#!/usr/bin/env python3

from __future__ import division
from __future__ import print_function

import numpy as np
import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QStatusBar,
    QPushButton, QLabel, QFrame, QListWidget)
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QBrush, QPixmap)
from PyQt5.QtCore import (
    Qt, pyqtSlot)

from bite_selection_package.config import spanet_config as config


class PyQtSamplerMainAxis(QMainWindow):
    def __init__(self, keyword="all", for_test=False):
        super(PyQtSamplerMainAxis, self).__init__()
        self.setMouseTracking(True)

        if for_test:
            self.base_dir = os.path.join(
                '../samples', 'food_spanet_rgb')
        else:
            self.base_dir = "../data/skewering_positions_%s" % (keyword)

        self.img_dir = os.path.join(self.base_dir, 'cropped_images')
        self.img_filename_list = list()
        self.img_idx = 0

        self.ann_dir = os.path.join(self.base_dir, 'annotations')

        self.cur_category = None

        self.label = None
        self.pixmap = None

        self.org_img_size = None
        self.label_size = 450
        self.pixmap_offset = None

        self.margin = 10
        self.btn_width = 80
        self.btn_height = 30
        self.statusbar_height = 20

        self.shift_pressed = False
        self.ctrl_pressed = False

        self.title = 'Main Axis Sampler'
        self.left = 150
        self.top = 100
        self.width = 840
        self.height = self.label_size + self.margin * 3 + \
            self.btn_height + self.statusbar_height

        self.init_data()
        self.init_ui()

    def init_data(self):
        print('init data')
        filename_list = os.listdir(self.img_dir)

        for item in sorted(filename_list):
            if not item.endswith('.png'):
                continue

            item_name = item.split('.')[0]

            self.img_filename_list.append(item_name)
        print('loaded {} cropped images'.format(len(self.img_filename_list)))

        if not os.path.exists(self.ann_dir):
            os.makedirs(self.ann_dir)

    def init_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setMinimumSize(350, 350)

        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QColor(238, 239, 236))
        self.setPalette(p)

        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        self.init_buttons()
        self.show_image()

        self.list_widget = QListWidget(self)
        self.list_widget.itemDoubleClicked.connect(self.on_item_double_click)
        self.update_list()
        self.update_list_data()

        self.show()

    def init_buttons(self):
        self.btn_prev_text = '<< Prev'
        self.btn_prev = QPushButton(self.btn_prev_text, self)
        self.btn_prev.clicked.connect(self.on_btn_prev_click)

        self.btn_save_text = 'Save'
        self.btn_save = QPushButton(self.btn_save_text, self)
        self.btn_save.clicked.connect(self.on_btn_save_click)

        self.btn_clear_text = 'Clear'
        self.btn_clear = QPushButton(self.btn_clear_text, self)
        self.btn_clear.clicked.connect(self.on_btn_clear_click)

        self.btn_next_text = 'Next >>'
        self.btn_next = QPushButton(self.btn_next_text, self)
        self.btn_next.clicked.connect(self.on_btn_next_click)

        self.adjust_buttons()

    def adjust_buttons(self):
        self.btn_width = self.label_size * 0.23

        self.btn_prev.move(
            self.margin, self.label_size + self.margin * 2)
        self.btn_prev.resize(self.btn_width, self.btn_height)

        self.btn_save.move(
            self.margin + self.label_size * 0.37 - self.btn_width * 0.5,
            self.label_size + self.margin * 2)
        self.btn_save.resize(self.btn_width, self.btn_height)

        self.btn_clear.move(
            self.margin + self.label_size * 0.63 - self.btn_width * 0.5,
            self.label_size + self.margin * 2)
        self.btn_clear.resize(self.btn_width, self.btn_height)

        self.btn_next.move(
            self.label_size - self.btn_width + self.margin,
            self.label_size + self.margin * 2)
        self.btn_next.resize(self.btn_width, self.btn_height)

    @pyqtSlot()
    def on_btn_prev_click(self):
        self.show_prev_image()

    @pyqtSlot()
    def on_btn_save_click(self):
        self.save_ann()

    @pyqtSlot()
    def on_btn_clear_click(self):
        self.clear_ann()

    @pyqtSlot()
    def on_btn_next_click(self):
        self.show_next_image()

    def show_prev_image(self):
        if self.img_idx > 0:
            self.img_idx -= 1
            self.list_widget.setCurrentRow(self.img_idx)
            self.show_image()
        else:
            print('This is the first image in the dataset')

    def show_next_image(self):
        if self.img_idx < len(self.img_filename_list) - 1:
            self.img_idx += 1
            self.list_widget.setCurrentRow(self.img_idx)
            self.show_image()
        else:
            print('This is the last image in the dataset')

    def update_list_data(self):
        self.list_widget.addItems(self.img_filename_list)

    def update_list(self):
        self.list_widget.move(self.label_size + self.margin * 2, self.margin)
        self.list_widget.resize(
            self.width - (self.label_size + self.margin * 3),
            self.height - (self.margin * 2 + self.statusbar_height))

    def on_item_double_click(self, item):
        self.img_idx = self.list_widget.currentRow()
        self.list_widget.clearFocus()
        self.show_image()

    def set_statusbar(self, optional=None):
        msg = 'Recording masks:  {} / {}   |   Image Directory: "{}"'.format(
                self.img_idx + 1, len(self.img_filename_list), self.img_dir)
        if optional is not None:
            msg += '  | {}'.format(optional)

        self.statusbar.showMessage(msg)

    def show_image(self):
        img_filename = self.img_filename_list[self.img_idx]
        self.cur_category = img_filename.split('_')[-2]
        filepath = os.path.join(
            self.img_dir, img_filename + '.png')

        self.pixmap = QPixmap(filepath)
        self.org_img_size = np.array(
            [self.pixmap.width(), self.pixmap.height()])
        self.rescale_image()

        if self.label is None:
            self.label = OverlayLabel(self)
            self.label.setAlignment(Qt.AlignCenter)
            self.label.setFrameShape(QFrame.Panel)
            self.label.setFrameShadow(QFrame.Sunken)
            self.label.setLineWidth(1)

        self.label.setPixmap(self.pixmap)
        self.label.move(self.margin, self.margin)
        self.label.resize(self.label_size, self.label_size)

        self.load_ann_from_file()

        self.set_statusbar()
        self.update()

    def resize_image(self):
        self.rescale_image()
        self.label.setPixmap(self.pixmap)
        self.label.move(self.margin, self.margin)
        self.label.resize(self.label_size, self.label_size)
        self.update_list()
        self.update()

    def rescale_image(self):
        ratio = self.label_size / float(np.max(self.org_img_size))
        new_size = self.org_img_size * ratio
        self.pixmap = self.pixmap.scaled(new_size[0], new_size[1])
        self.pixmap_offset = (self.label_size - new_size) * 0.5 + self.margin

    def calculate_angle(self, sp=None, ep=None):
        if sp is None:
            sp = self.label.axis_sp
        if ep is None:
            ep = self.label.axis_ep
        pdiff = sp - ep
        return np.degrees(
            np.arctan2(pdiff[0], pdiff[1])) % 180

    def get_center_of_highlight(self):
        targets = list()
        highlight = self.label.highlight
        for ci in range(highlight.shape[0]):
            for ri in range(highlight.shape[1]):
                if highlight[ri, ci] == 1:
                    targets.append([ri, ci])
        targets = np.asarray(targets)
        return np.sum(targets / targets.shape[0], axis=0)

    def add_group_item(self, x, y):
        if x < 0 or x >= self.grid_size[0] or y < 0 or y >= self.grid_size[1]:
            return False
        if self.label.grid[x, y] == -1:
            return False

        if self.cur_group is None:
            self.cur_group = dict()

        hv = x * self.grid_size[0] + y

        if hv in self.cur_group:
            return False
        self.cur_group[hv] = True
        return True

    def clear_ann(self):
        self.label.clear_ann()
        self.update()

    def save_ann(self):
        if (self.label.axis_sp is None or len(self.label.axis_sp) == 0 or
                self.label.axis_ep is None or len(self.label.axis_ep) == 0):
            return

        if not os.path.exists(self.ann_dir):
            os.makedirs(self.ann_dir)
        img_name = self.img_filename_list[self.img_idx]
        ann_filename = os.path.join(
            self.ann_dir, img_name) + '.txt'

        with open(ann_filename, 'w') as f:
            this_sp = (np.array(self.label.axis_sp) - self.margin) / self.label_size
            this_ep = (np.array(self.label.axis_ep) - self.margin) / self.label_size

            f.write('{0:.3f} {1:.3f} {2:.3f} {3:.3f}\n'.format(
                *this_sp, *this_ep))
            f.close()
            print('saved: {}'.format(ann_filename))

    def load_ann_from_file(self):
        img_name = self.img_filename_list[self.img_idx]
        ann_filename = os.path.join(
            self.ann_dir, img_name) + '.txt'

        self.clear_ann()
        if not os.path.exists(ann_filename):
            return

        with open(ann_filename, 'r') as f:
            points = np.array(list(map(float, f.read().strip().split())))
            self.label.axis_sp = points[:2] * self.label_size + self.margin
            self.label.axis_ep = points[2:] * self.label_size + self.margin
            f.close()

    def show_shortcuts(self, show=True):
        if show:
            self.btn_prev.setText('<< A')
            self.btn_save.setText('S')
            self.btn_clear.setText('C')
            self.btn_next.setText('D >>')
        else:
            self.btn_prev.setText(self.btn_prev_text)
            self.btn_save.setText(self.btn_save_text)
            self.btn_clear.setText(self.btn_clear_text)
            self.btn_next.setText(self.btn_next_text)

    # override
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_A:
            self.show_prev_image()
        elif event.key() == Qt.Key_S:
            self.save_ann()
        elif event.key() == Qt.Key_C:
            self.clear_ann()
        elif event.key() == Qt.Key_D:
            self.save_ann()
            self.show_next_image()
        elif event.key() == Qt.Key_Alt:
            self.show_shortcuts()
        elif event.key() == Qt.Key_Shift:
            self.shift_pressed = True
        elif event.key() == Qt.Key_Control:
            self.ctrl_pressed = True

    # override
    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Alt:
            self.show_shortcuts(show=False)
        elif event.key() == Qt.Key_Shift:
            self.shift_pressed = False
        elif event.key() == Qt.Key_Control:
            self.ctrl_pressed = False

    # override
    def resizeEvent(self, event):
        self.width, self.height = event.size().width(), event.size().height()

        self.label_size = (
            self.height - (self.margin * 3 +
                           self.btn_height + self.statusbar_height))
        self.resize_image()

        self.adjust_buttons()

    # override
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.label.set_axis_ep(
                event.pos().x() - self.margin,
                event.pos().y() - self.margin)
            self.update()
        elif event.buttons() == Qt.RightButton:
            self.label.set_axis_ep(
                event.pos().x() - self.margin,
                event.pos().y() - self.margin)
            self.update()

    # override
    def mousePressEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.label.set_axis_sp(
                event.pos().x() - self.margin,
                event.pos().y() - self.margin)
        elif event.buttons() == Qt.RightButton:
            self.label.set_axis_sp(
                event.pos().x() - self.margin,
                event.pos().y() - self.margin)
        self.update()

    # override
    def mouseReleaseEvent(self, event):
        self.label.set_axis_ep(
            event.pos().x() - self.margin,
            event.pos().y() - self.margin)
        self.update()


class OverlayLabel(QLabel):
    def __init__(self, parent=None):
        super(OverlayLabel, self).__init__(parent=parent)
        self.axis_sp = None
        self.axis_ep = None

    def clear_ann(self):
        self.axis_sp = None
        self.axis_ep = None

    def set_axis_sp(self, x, y):
        self.axis_sp = np.array([x, y])

    def set_axis_ep(self, x, y):
        self.axis_ep = np.array([x, y])

    # override
    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)

        pen = QPen(QColor(255, 255, 255, alpha=130))
        pen.setCapStyle(Qt.RoundCap)
        pen.setWidth(1)

        pen_ang = QPen(QColor(255, 176, 59, alpha=230))
        pen_ang.setCapStyle(Qt.RoundCap)
        pen_ang.setWidth(2)

        pen_ang_grid = QPen(QColor(142, 40, 0, alpha=220))
        pen_ang_grid.setCapStyle(Qt.RoundCap)
        pen_ang_grid.setWidth(3)

        pen_ang_ch = QPen(QColor(182, 73, 38, alpha=230))
        pen_ang_ch.setCapStyle(Qt.RoundCap)
        pen_ang_ch.setWidth(2)

        brush = QBrush(QColor(255, 240, 165, alpha=120))
        brush.setStyle(Qt.SolidPattern)

        brush_hl = QBrush(QColor(70, 137, 102, alpha=180))
        brush_hl.setStyle(Qt.Dense1Pattern)

        painter.setBrush(Qt.NoBrush)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.Antialiasing, False)

        if (self.axis_sp is not None and len(self.axis_sp) > 0 and
                self.axis_ep is not None and len(self.axis_ep) > 0):
            painter.setBrush(Qt.NoBrush)
            painter.setPen(pen_ang)
            painter.setRenderHint(QPainter.Antialiasing, True)

            painter.drawLine(
                self.axis_sp[0], self.axis_sp[1],
                self.axis_ep[0], self.axis_ep[1])

            ch_len = 5
            painter.setPen(pen_ang_ch)
            painter.drawLine(
                self.axis_sp[0] - ch_len, self.axis_sp[1],
                self.axis_sp[0] + ch_len, self.axis_sp[1])
            painter.drawLine(
                self.axis_sp[0], self.axis_sp[1] - ch_len,
                self.axis_sp[0], self.axis_sp[1] + ch_len)

            painter.drawLine(
                self.axis_ep[0] - ch_len, self.axis_ep[1],
                self.axis_ep[0] + ch_len, self.axis_ep[1])
            painter.drawLine(
                self.axis_ep[0], self.axis_ep[1] - ch_len,
                self.axis_ep[0], self.axis_ep[1] + ch_len)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    if len(sys.argv) > 1:
    	ex = PyQtSamplerMainAxis(sys.argv[1])
    else:
    	ex = PyQtSamplerMainAxis()
    sys.exit(app.exec_())
