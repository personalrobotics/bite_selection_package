#!/usr/bin/env python

import xml.etree.ElementTree as ET
import os


def update_xml(filepath, filename, is_edit=False):
    print('>>> {}'.format(filename))
    tree = ET.parse(filepath)
    root = tree.getroot()

    for node in root:
        if node.tag == 'object':
            name_node = node.find('name')
            new_name = name_node.text.replace(' ', '_')

            # print('{} -> {}'.format(
            #     name_node.text,
            #     new_name))
            name_node.text = new_name

            if name_node.text == 'plate':
                print('plate!!')
                root.remove(node)

    if is_edit:
        tree.write(filepath)


def update_all_xmls(base_dir, is_edit=False):
    filenames = os.listdir(base_dir)

    trainval_list = list()

    for filename in filenames:
        filepath = os.path.join(base_dir, filename)
        update_xml(filepath, filename, is_edit)
        trainval_list.append(filename[:-4])

    f_tv = open('trainval.txt.new', 'w')
    for trainval in trainval_list:
        f_tv.write('%s\n' % trainval)
    f_tv.close()


def script_main():
    print('update_path')

    import sys
    is_edit = False
    if (len(sys.argv) == 2) and (sys.argv[1] == 'true'):
        is_edit = True
    update_all_xmls('xmls', is_edit=is_edit)

    print('edit mode: ', is_edit)


if __name__ == '__main__':
    script_main()


# End of script

