#!/bin/bash

cd ./scripts

catkin build bite_selection_package

source $(catkin locate)/devel/setup.bash

python3 ./spnet_train.py $1

cd -

