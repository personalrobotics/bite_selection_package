#!/bin/bash

python_version=`cat ./config/python_version`

python3 ./external_apps/labelImg/labelImg.py \
    ./data/bounding_boxes_c9/images \
    ./config/predefined_classes_foods_c9.txt \
    ./data/bounding_boxes_c9/annotations/xmls

