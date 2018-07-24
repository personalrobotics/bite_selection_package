# bite_selection_package

## Dependencies
This project uses `numpy`, `matplotlib`, `opencv`, `lxml` and `pyqt`. The setup script below will automatically install `lxml` and `pyqt`, but please check if you have `opencv` and `matplotlib` before you run skewering position sampler.

This project supports Linux and MacOS. If you are using Windows, please do not use helper scripts, and setup and run each program manually.


## Generating 2D bounding boxes by using labelImg
To build labelImg, run `setup_labelImg.sh` with a python version (python2 or python3) you want to use. For example,
```
./setup_labelImg.sh python3
```
It will install pyqt-dev-tools and python-lxml, and build labelImg.

If you successfully built labelImg, you can start it by typing:
```
./run_labelImg.sh
```

All the images are in `data/bounding_boxes/images`, and annotations will be saved in `data/bounding_boxes/annotations/xmls`.


## Generating skewering positions and rotations
To record skewering positions, run `run_skewering_position_sampler.sh` with an option:
```
./run_skewering_position_sampler.sh <option>
```
Available options are:
- `all`: running both cropping and sampling
- `crop`: generating cropped images from images and annotations in `data/bounding_boxes`
- `sample`: sampling skewering position and rotation for each cropped image


## SPNet: CNNs for Estimating Skewering Positions

### Preprocess Skewering Dataset
```
./preprocess_spdataset.sh
```
This script will resize and pad all the images in `data/skewering_positions/` and save them in `data/processed/`.

### Training SPNet
```
./spnet_train.sh
```
The training script will train `SPNet` with the cropped images and annotations in `data/processed/` and save its checkpoint file as `checkpoints/spnet_ckpt.pth`.
