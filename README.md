# bite_selection_package

## Dependencies
This project uses `catkin`, `numpy`, `matplotlib`, `opencv`, `lxml` and `pyqt5`. The setup script below will automatically install `lxml` and `pyqt5`, but please check if you have `opencv` and `matplotlib` before you run skewering position sampler.

Labeling part of this project supports Linux and MacOS. If you are using Windows, please do not use helper scripts, and setup and run each program manually.

In order to use the neural network models, SPNet or SPANet, please clone this project in the catkin workspace and build before you run training or test scripts.


## Installation
```
cd YOUR_CATKIN_WS/src
git clone https://github.com/personalrobotics/bite_selection_package.git
cd ./bite_selection_package
./load_checkpoint.sh
catkin build bite_selection_package
source $(catkin locate)/devel/setup.bash
```

To test SPNet, please run the tutorial script, `examples/spnet_tutorial.py`:
```
cd ./examples
./spnet_tutorial.py
```
<img src="https://github.com/personalrobotics/bite_selection_package/blob/master/examples/test_result.png?raw=true" width="450">

## Collecting images
To collect images for training, please check this [image collection script](https://github.com/personalrobotics/image_collector).


## Generating 2D bounding boxes by using labelImg (RetinaNet)
To build labelImg, run `setup_labelImg.sh`. **Note**: Labeling tools in this project only support python3 with PyQt5. Please use caution before running this script if you are using python2 + pyqt4 on your system.
```
./setup_labelImg.sh
```
It will install python3-dev, python3-pip, pyqt5-dev-tools, and lxml for python3 and build labelImg.

If you successfully built labelImg, you can start it by typing:
```
./run_labelImg.sh
```

All the images are in `data/bounding_boxes/images`, and annotations will be saved in `data/bounding_boxes/annotations/xmls`.

Please check this [labelImg](https://github.com/personalrobotics/labelImg) repo if you want to modify the annotation tool.

You can also use other annotation tools. Here are some suggestions:
* http://labelme2.csail.mit.edu
* https://rectlabel.com/
* http://www.cs.toronto.edu/polyrnn/


### Generating 2D bounding boxes in Unreal Engine
* https://github.com/personalrobotics/unrealcv
* https://github.com/personalrobotics/UnrealCV_ROS


## Generating skewering positions and rotations
To generate cropped images for training SPNet or SPANet, run `skewering_position_sampler`:
```
cd ./scripts
./generate_cropped_images.py <keyword>
```
This script will generate cropped images from images and annotations in `data/bounding_boxes_<keyword>` and save them in `data/skewering_positions_<keyword>`.

### SPNet
To generate mask annotations for SPNet, use `PyQtSampler`:
```
cd ./scripts
./qt_sampler.py
```

### SPANet
To generate main axis annotations for SPANet, use this separate `PyQtSampler`:
```
cd ./scripts
./qt_sampler_main_axis.py <keyword>
```

## RetinaNet: Object Detection Network
We used [RetinaNet](https://github.com/personalrobotics/pytorch_retinanet) for object detection. After you make a symlink of a new dataset with images and bounding boxes into `pytorch_retinanet/data/`, you can train RetinaNet with the dataset.


## SPNet: CNNs for Estimating Skewering Positions

### Training SPNet/SPANet
```
./<spnet/spanet>_train.sh
```
The training script will train `SPNet` or `SPAnet` with the cropped images and annotations in the directories specified in `src/bite_selection_package/config/<spnet/spanet>_config.py` and save its checkpoint file as `checkpoints/food_spnet_<keyword>_ckpt.pth`.

