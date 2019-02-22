#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
scp prl@mthrbrn.personalrobotics.cs.washington.edu:/mnt/hard_data/Data/foods/bite_selection_package/data/food_label_map.pbtxt $DIR
scp prl@mthrbrn.personalrobotics.cs.washington.edu:/mnt/hard_data/Data/foods/bite_selection_package/data/food_general_label_map.pbtxt $DIR

