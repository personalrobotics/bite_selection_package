#!/bin/bash

python_version=`cat ./config/python_version`

cd ./scripts

$python_version ./preprocess_cropped_images.py

cd -
