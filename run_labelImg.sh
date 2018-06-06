#!/bin/bash

python_version=`cat ./config/python_version`

$python_version ./external_apps/labelImg/labelImg.py \
    ./data/bounding_boxes/images \
    ./external_apps/labelImg/data/predefined_classes.txt \
    ./data/bounding_boxes/annotations/xmls

