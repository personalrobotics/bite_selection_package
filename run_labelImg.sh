#!/bin/bash

python_version=`cat ./config/python_version`

python3 ./external_apps/labelImg/labelImg.py \
    ./data/bounding_boxes_c7/images \
    ./config/predefined_classes_foods_c7.txt \
    ./data/bounding_boxes_c7/annotations/xmls

