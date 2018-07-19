#!/bin/bash

python_version=`cat ./config/python_version`

cd ./scripts

$python_version ./skewering_position_sampler.py view ../samples

cd -
