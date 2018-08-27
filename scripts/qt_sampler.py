#!/usr/bin/env python3

from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QStatusBar,
    QPushButton, QLabel, QFrame, QListWidget)
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QBrush, QPixmap)
from PyQt5.QtCore import (
    Qt, pyqtSlot)


class PyQtSampler(QMainWindow):
    def __init__(self):
        super(PyQtSampler, self).__init__()
        self.setMouseTracking(True)

        self.base_dir = '../data/skewering_positions'

        self.img_dir = os.path.join(self.base_dir, 'cropped_images')
        self.img_filename_list = list()
        self.img_idx = 0

        self.mask_dir = os.path.join(self.base_dir, 'masks')

        self.category_types = {
            "apple": 'flat',
            "apricot": 'flat',
            "banana": 'any',
            "pepper": 'flat',
            "blackberry": 'any',
            "broccoli": 'flat',
            "cantalope": 'flat',
            "carrot": 'flat',
            "cauliflower": 'flat',
            "celery": 'flat',
            "tomato": 'any',
            "egg": 'round',
            "grape": 'any',
            "green": 'any',
            "purple": 'any',
            "melon": 'flat',
            "strawberry": 'flat',
            "plate": 'none', }
        self.cur_category = None

        self.label = None
        self.pixmap = None
        # self.grid_size = (17, 17)
        self.grid_size = (11, 11)

        self.cur_group = None

        self.org_img_size = None
        self.label_size = 450
        self.pixmap_offset = None

        self.margin = 10
        self.btn_width = 80
        self.btn_height = 30
        self.statusbar_height = 20

        self.shift_pressed = False
        self.ctrl_pressed = False

        self.title = 'Skewering Mask Sampler'
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
            if not item.endswith('.jpg'):
                continue

            item_name = item.split('.')[0]
            this_category = item_name.split('_')[-2]
            if self.category_types[this_category] == 'none':
                continue

            self.img_filename_list.append(item_name)
        print('loaded {} cropped images'.format(len(self.img_filename_list)))

        if not os.path.exists(self.mask_dir):
            os.makedirs(self.mask_dir)

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
        self.save_grid()

    @pyqtSlot()
    def on_btn_clear_click(self):
        self.clear_grid()

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
            self.img_dir, img_filename + '.jpg')

        self.pixmap = QPixmap(filepath)
        self.org_img_size = np.array(
            [self.pixmap.width(), self.pixmap.height()])
        self.rescale_image()

        if self.label is None:
            self.label = OverlayLabel(self, grid_size=self.grid_size)
            self.label.setAlignment(Qt.AlignCenter)
            self.label.setFrameShape(QFrame.Panel)
            self.label.setFrameShadow(QFrame.Sunken)
            self.label.setLineWidth(1)

        self.label.setPixmap(self.pixmap)
        self.label.move(self.margin, self.margin)
        self.label.resize(self.label_size, self.label_size)

        self.load_grid_from_file()

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
            sp = self.label.angle_sp
        if ep is None:
            ep = self.label.angle_ep
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

    def propagate_selection(self, x, y):
        next_steps = [
            [x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1],
            [x + 1, y + 1], [x - 1, y - 1], [x - 1, y + 1], [x + 1, y - 1]]
        for item in next_steps:
            if self.add_group_item(item[0], item[1]):
                self.propagate_selection(item[0], item[1])

    def select_group(self, pos):
        grid_idx = self.pos_to_grid_idx(pos)
        if grid_idx is None or self.label.grid[grid_idx[0], grid_idx[1]] == -1:
            self.clear_group()
            return

        self.add_group_item(grid_idx[0], grid_idx[1])
        self.propagate_selection(grid_idx[0], grid_idx[1])

        for k, v in sorted(self.cur_group.items()):
            this_idx = [k // self.grid_size[0], k % self.grid_size[0]]
            self.label.set_highlight_at(this_idx[0], this_idx[1], 1)

    def clear_group(self):
        self.label.clear_highlight()
        self.cur_group = None

    def pos_to_grid_idx(self, pos):
        ratio_in_label = (
            (np.array([pos.x(), pos.y()]) - self.margin) / self.label_size)
        if np.min(ratio_in_label) < 0 or np.max(ratio_in_label) >= 1:
            return None
        return list(map(int, ratio_in_label * self.grid_size))

    def grid_idx_to_pos(self, grid_idx):
        return (np.asarray(grid_idx) * self.label_size / self.grid_size +
                self.margin)

    def update_grid_by_pos(self, pos, val=0, is_left=True):
        grid_idx = self.pos_to_grid_idx(pos)
        if grid_idx is None:
            return
        self.label.set_grid_at(
            grid_idx[0], grid_idx[1],
            val if is_left and not self.ctrl_pressed else -1)

    def update_highlighted_grids(self, angle):
        highlight = self.label.highlight
        for ci in range(highlight.shape[0]):
            for ri in range(highlight.shape[1]):
                if highlight[ri, ci] == 1:
                    if self.category_types[self.cur_category] == 'flat':
                        self.label.set_grid_at(ri, ci, angle)
                    else:
                        this_sp = self.grid_idx_to_pos([ri, ci])
                        this_angle = self.calculate_angle(sp=this_sp)
                        self.label.set_grid_at(ri, ci, this_angle)

    def clear_grid(self):
        self.label.clear_grid()
        self.update()

    def save_grid(self):
        if not os.path.exists(self.mask_dir):
            os.makedirs(self.mask_dir)
        img_name = self.img_filename_list[self.img_idx]
        mask_filename = os.path.join(
            self.mask_dir, img_name) + '.txt'

        grid = self.label.grid
        with open(mask_filename, 'w') as f:
            for ci in range(grid.shape[1]):
                for ri in range(grid.shape[0]):
                    f.write('{0:.1f}'.format(grid[ri, ci]))
                    if ri + 1 < grid.shape[1]:
                        f.write(',')
                f.write('\n')
            f.close()
            print('saved: {}'.format(mask_filename))

    def resize_grid(self, grid, new_size):
        new_grid = np.zeros(new_size)
        nw, nh = new_size
        ow, oh = grid.shape
        for wi in range(new_size[0]):
            for hi in range(new_size[1]):
                new_grid[wi, hi] = grid[
                    int(wi / nw * ow),
                    int(hi / nh * oh)]
        return new_grid

    def load_grid_from_file(self):
        img_name = self.img_filename_list[self.img_idx]
        mask_filename = os.path.join(
            self.mask_dir, img_name) + '.txt'

        self.clear_grid()
        if not os.path.exists(mask_filename):
            return

        new_grid = list()
        with open(mask_filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split(',')
                new_row = list()
                for item in items:
                    new_row.append(float(item.strip()))
                new_grid.append(new_row)
            f.close()

        new_grid = np.asarray(new_grid).transpose()
        new_grid = self.resize_grid(new_grid, self.grid_size)
        self.label.grid = new_grid

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
            self.save_grid()
        elif event.key() == Qt.Key_C:
            self.clear_grid()
        elif event.key() == Qt.Key_D:
            self.save_grid()
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
            if self.shift_pressed:
                self.label.set_angle_ep(
                    event.pos().x() - self.margin,
                    event.pos().y() - self.margin)
                angle = self.calculate_angle()
                self.update_highlighted_grids(angle)
                self.set_statusbar('angle: {0:.3f}'.format(angle))
            else:
                self.update_grid_by_pos(event.pos(), is_left=True)
            self.update()
        elif event.buttons() == Qt.RightButton:
            self.update_grid_by_pos(event.pos(), is_left=False)
            self.update()

    # override
    def mousePressEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            if self.shift_pressed:
                self.label.set_angle_sp(
                    event.pos().x() - self.margin,
                    event.pos().y() - self.margin,
                    angle_type=self.category_types[self.cur_category])
                self.select_group(event.pos())
            else:
                self.update_grid_by_pos(event.pos(), is_left=True)
        elif event.buttons() == Qt.RightButton:
            self.update_grid_by_pos(event.pos(), is_left=False)
        self.update()

    # override
    def mouseReleaseEvent(self, event):
        if self.label.angle_sp is not None and self.label.angle_ep is not None:
            angle = self.calculate_angle()
            print('angle for this group: {}'.format(angle))

        self.clear_group()
        self.label.angle_sp = None
        self.label.angle_ep = None
        self.set_statusbar()
        self.update()


class OverlayLabel(QLabel):
    def __init__(self, parent=None, grid_size=(17, 17)):
        super(OverlayLabel, self).__init__(parent=parent)
        self.grid_size = grid_size
        self.grid = None
        self.highlight = None
        self.angle_sp = None
        self.angle_ep = None
        self.angle_type = 'flat'

    def init_grid(self):
        self.grid = np.ones(self.grid_size, dtype=np.float) * -1

    def set_grid_at(self, x, y, v=0.):
        self.grid[x, y] = v

    def clear_grid(self):
        if self.grid is None:
            self.init_grid()
        else:
            self.grid[:] = -1

    def init_highlight(self):
        self.highlight = np.zeros(self.grid_size, dtype=np.int32)

    def set_highlight_at(self, x, y, v=1):
        self.highlight[x, y] = v

    def clear_highlight(self):
        if self.highlight is None:
            self.init_highlight()
        else:
            self.highlight[:] = 0

    def set_angle_sp(self, x, y, angle_type='flat'):
        self.angle_sp = np.array([x, y])
        self.angle_type = angle_type

    def clear_angle_sp(self):
        self.angle_sp = None

    def set_angle_ep(self, x, y):
        self.angle_ep = np.array([x, y])

    def clear_angle_ep(self):
        self.angle_ep = None

    # override
    def paintEvent(self, event):
        super().paintEvent(event)

        if self.grid is None:
            self.init_grid()

        if self.highlight is None:
            self.init_highlight()

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

        grid_w = self.width() / self.grid.shape[0]
        grid_h = self.height() / self.grid.shape[1]

        # draw grid lines
        for wi in range(1, self.grid.shape[0]):
            painter.drawLine(
                grid_w * wi, 0,
                grid_w * wi, self.width())

        for hi in range(1, self.grid.shape[1]):
            painter.drawLine(
                0, grid_h * hi,
                self.height(), grid_h * hi)

        painter.setRenderHint(QPainter.Antialiasing, False)

        rho = min(grid_w, grid_h) * 0.3
        for wi in range(self.grid.shape[0]):
            for hi in range(self.grid.shape[1]):
                grid_val = self.grid[wi, hi]
                if grid_val > -1:
                    # draw selected grid and highlight
                    painter.setPen(Qt.NoPen)
                    if self.highlight[wi, hi] == 1:
                        painter.setBrush(brush_hl)
                    else:
                        painter.setBrush(brush)

                    painter.drawRect(
                        grid_w * wi + 1, grid_h * hi + 1,
                        grid_w - 1, grid_h - 1)

                    # draw saved angles
                    painter.setPen(pen_ang_grid)
                    painter.setBrush(Qt.NoBrush)

                    grid_cp = np.array([
                        grid_w * wi + grid_w * 0.5,
                        grid_h * hi + grid_h * 0.5])
                    gdiff = np.array([
                        np.sin(np.radians(grid_val)),
                        np.cos(np.radians(grid_val))]) * rho
                    grid_angle_sp = grid_cp - gdiff
                    grid_angle_ep = grid_cp + gdiff
                    painter.drawLine(
                        grid_angle_sp[0], grid_angle_sp[1],
                        grid_angle_ep[0], grid_angle_ep[1])

        if self.angle_sp is not None and self.angle_ep is not None:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(pen_ang)
            painter.setRenderHint(QPainter.Antialiasing, True)

            if self.angle_type == 'flat':
                popp = self.angle_sp + (self.angle_sp - self.angle_ep)
                painter.drawLine(
                    popp[0], popp[1],
                    self.angle_ep[0], self.angle_ep[1])

            painter.setPen(pen_ang_ch)
            painter.drawLine(
                self.angle_ep[0] - rho, self.angle_ep[1],
                self.angle_ep[0] + rho, self.angle_ep[1])
            painter.drawLine(
                self.angle_ep[0], self.angle_ep[1] - rho,
                self.angle_ep[0], self.angle_ep[1] + rho)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PyQtSampler()
    sys.exit(app.exec_())
