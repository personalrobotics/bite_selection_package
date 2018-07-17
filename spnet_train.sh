#!/bin/bash

python_version=`cat ./config/python_version`

cd ./scripts

$python_version ./spnet_train.py $1

cd -

