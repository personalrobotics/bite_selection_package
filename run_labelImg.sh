#!/bin/bash

python_version=`cat ./config/python_version`

python3 ./external_apps/labelImg/labelImg.py \
    ./data/bounding_boxes/images \
    ./config/predefined_classes_foods.txt \
    ./data/bounding_boxes/annotations/xmls

