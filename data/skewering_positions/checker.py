#!/usr/bin/env python

import os


def checker():
    print('checker')

    base_dir = './masks'
    filenames = os.listdir(base_dir)

    for fn in filenames:
        if not fn.endswith('.txt'):
            continue
        with open(os.path.join(base_dir, fn), 'r') as f:
            ln = f.readlines()
            cn = ln[0].split(',')
            if len(ln) != 17 or len(cn) != 17:
                print(len(ln), len(cn))
            f.close()


if __name__ == '__main__':
    checker()
