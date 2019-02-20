#!/bin/bash

# python_version=`cat ./config/python_version`

python3 ./external_apps/labelImg/labelImg.py \
    ~/external/Data/food_manipulation/data_collection/bounding_boxes_general/images \
    ./config/predefined_classes_foods_general.txt \
    ~/external/Data/food_manipulation/data_collection/bounding_boxes_general/annotations/xmls

