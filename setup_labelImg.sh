#!/bin/bash

print_usage() {
  echo "please specify python version you want to use"
  echo "usage:"
  echo "    ./setup_labelImg.sh python3 or (python2)"
}

if [ -d "./config" ]; then
  rm -rf ./config
fi

git submodule update --init --recursive

if [ "$OSTYPE" == "linux-gnu" ]; then
  if [ "$1" == "python3" ]; then
    echo "installing labelImg with Python3 + Qt5"
    cd ./external_apps/labelImg
    sudo apt install python3-dev python3-pip pyqt5-dev-tools && \
        sudo pip3 install lxml && \
        make qt5py3
    cd ../../
    mkdir config && cd config && echo "python3" > python_version && cd ../

  elif [ "$1" == "python2" ]; then
    echo "installing labelImg with Python2 + Qt4"
    cd ./external_apps/labelImg
    sudo apt install python-dev python-pip pyqt4-dev-tools && \
      sudo pip2 install lxml && \
      make qt4py2
    cd ../../
    mkdir config && cd config && echo "python2" > python_version && cd ../

  else
    print_usage
  fi

elif [ "$OSTYPE" == "darwin"* ]; then
  if [ "$1" == "python3" ]; then
    echo "installing labelImg with Python3 + Qt5"
    cd ./external_apps/labelImg
    brew install qt && \
        brew install libxml2 && \
        pip3 install PyQt5 lxml && \
        make qt5py3
    cd ../../
    mkdir config && cd config && echo "python3" > python_version && cd ../

  elif [ "$1" == "python2" ]; then
    echo "installing labelImg with Python2 + Qt4"
    cd ./external_apps/labelImg
    brew install qt4 && \
        brew install libxml2 && \
        pip2 install PyQt4 lxml && \
        make qt4py2
    cd ../../
    mkdir config && cd config && echo "python2" > python_version && cd ../

  else
    print_usage
  fi

else
  echo "This script does not support \"${OSTYPE}\"."
fi

