#!/bin/bash

python_version=`cat ./config/python_version`

python3 ./external_apps/labelImg/labelImg.py \
    ./data/bounding_boxes_c8/images \
    ./config/predefined_classes_foods_c8.txt \
    ./data/bounding_boxes_c8/annotations/xmls

