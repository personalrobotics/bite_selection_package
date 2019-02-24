#!/bin/bash


git submodule update --init --recursive

uname_out="$(uname -s)"
case "${uname_out}" in
    Linux*)
      echo "installing labelImg with Python3 + Qt5"
      cd ./external_apps/labelImg
      sudo apt install python3-dev python3-pip pyqt5-dev-tools && \
          python3 -m pip install lxml && \
          make qt5py3
      cd -
      ;;

    Darwin*)
      echo "installing labelImg with Python3 + Qt5"
      cd ./external_apps/labelImg
      brew install qt && \
          brew install libxml2 && \
          python3 -m pip install PyQt5 lxml && \
          make qt5py3
      cd -
      ;;

    *)
      echo "This script does not support \"${uname_out}\"."
      ;;
esac

