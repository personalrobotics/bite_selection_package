#!/bin/bash

if [ "$1" == "python3" ]; then
  echo "installing labelImg with Python3 + Qt5"
  git submodule update --init --recursive
  cd ./external_apps/labelImg
  sudo apt install pyqt5-dev-tools && \
      sudo pip3 install lxml && \
      make qt5py3
  cd ../../
  mkdir config && cd config && echo "python3" > python_version && cd ../

elif [ "$1" == "python2" ]; then
  echo "installing labelImg with Python2 + Qt4"
  git submodule update --init --recursive
  cd ./external_apps/labelImg
  sudo apt install pyqt4-dev-tools && \
    sudo pip2 install lxml && \
    make qt4py2
  cd ../../
  mkdir config && cd config && echo "python2" > python_version && cd ../

else
  echo "please specify python version you want to use"
  echo "usage:"
  echo "    ./setup_labelImg.sh python3 or (python2)"
fi

