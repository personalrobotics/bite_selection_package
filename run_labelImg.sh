#!/bin/bash

python3 ./external_apps/labelImg/labelImg.py \
    ./data/bounding_boxes_spnet_all/images \
    ./src/bite_selection_package/config/predefined_classes_foods.txt \
    ./data/bounding_boxes_spnet_all/annotations/xmls

