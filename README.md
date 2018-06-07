# bite_selection_package

## Making 2D bounding boxes by using labelImg
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

## Recording skewering positions
To record skewering positions, run `run_skewering_position_sampler.sh` with an option:
```
./run_skewering_position_sampler.sh <option>
```
Available options are:
- `all`: running both cropping and sampling
- `crop`: generating cropped images from images and annotations in `data/bounding_boxes`
- `sample`: sampling skewering position and rotation for each cropped image

