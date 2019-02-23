#!/usr/bin/env python3

import numpy as np
import os


def clean_up_logs(basedir, filename):
    print('clean_up_logs')

    if not os.path.exists(basedir) or not os.path.isdir(basedir):
        print('invalid basedir: {}'.format(basedir))
        return

    src_path = os.path.join(basedir, filename)
    new_path = os.path.join(basedir, 'clean_{}'.format(filename))

    if not os.path.exists(src_path):
        print('cannot find: {}'.format(src_path))
        return

    with open(src_path, 'r') as src_f:
        with open(new_path, 'w') as new_f:
            lines = src_f.readlines()

            avg_values = np.zeros(8)
            avg_steps = 100

            title_line = 'epoch, loss, accuracy, precision, recall, f1_score, rotation'
            new_f.write(title_line + '\n')
            print(title_line)
            for i in range(len(lines)):
                values = np.asarray(list(map(float, lines[i].strip().split(','))))
                avg_values += values

                if i > 0 and i % avg_steps == 0:
                    loss = avg_values[2] / avg_steps / 4.25
                    rot_error = 1 - (avg_values[7] / avg_steps / 90.)
                    new_line = '{2:.0f}, {0:.3f}, {5:.3f}, {6:.3f}, {7:.3f}, {8:.3f}, {1:.3f}'.format(
                        loss, rot_error, *(avg_values / avg_steps))
                    new_f.write(new_line + '\n')
                    print(new_line)
                    avg_values[:] = 0

            src_f.close()
            new_f.close()


if __name__ == '__main__':
    clean_up_logs('food_spnet_c7_dense_3_6_a_18', 'training.txt')
    clean_up_logs('food_spnet_c7_dense_3_6_a_18', 'test.txt')
    clean_up_logs('food_spnet_fl_c7_dense_3_6_a_18', 'training.txt')
    clean_up_logs('food_spnet_fl_c7_dense_3_6_a_18', 'test.txt')