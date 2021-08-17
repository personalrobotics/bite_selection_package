#!/bin/bash

echo "load spnet checkpoints from mthrbrn"
if [ ! -d "./checkpoint" ]; then
  mkdir ./checkpoint
fi
scp -r prl@mthrbrn.personalrobotics.cs.washington.edu:/mnt/hard_data/Checkpoints/public/spnet_ckpt.pth ./checkpoint/

echo "load spanet checkpoints from mthrbrn"
scp -r prl@mthrbrn.personalrobotics.cs.washington.edu:/mnt/hard_data/Checkpoints/public/spanet_ckpt.pth ./checkpoint/
scp -r prl@mthrbrn.personalrobotics.cs.washington.edu:/mnt/hard_data/Checkpoints/public/spanet_wall.pth ./checkpoint/

echo "load label_map definition from mthrbrn"
if [ ! -d "./data" ]; then
  mkdir ./data
fi
scp prl@mthrbrn.personalrobotics.cs.washington.edu:/mnt/hard_data/Data/foods/bite_selection_package/food_detector/food_label_map.pbtxt ./data/

