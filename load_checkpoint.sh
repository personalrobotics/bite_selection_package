#!/bin/bash

echo "load spnet checkpoints from mthrbrn"
if [ ! -d "./checkpoint" ]; then
  mkdir ./checkpoint
fi
scp -r prl@mthrbrn.personalrobotics.cs.washington.edu:/mnt/hard_data/Checkpoints/bite_selection_package/checkpoint/spnet_ckpt.pth ./checkpoint/

echo "load spanet checkpoints from mthrbrn"
scp prl@mthrbrn.personalrobotics.cs.washington.edu:/home/ekgordon/Workspace/isrr_ws/src/bite_selection_package/checkpoint/food_spanet_all_rgb_wall_ckpt_best.pth ./checkpoint/spanet_with_wall_ckpt.pth

echo "load label_map definition from mthrbrn"
if [ ! -d "./data" ]; then
  mkdir ./data
fi
scp prl@mthrbrn.personalrobotics.cs.washington.edu:/mnt/hard_data/Data/foods/bite_selection_package/food_detector/food_label_map.pbtxt ./data/

