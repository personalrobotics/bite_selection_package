#!/bin/bash

print_usage() {
  echo "usage:"
  echo "    ./run_skewering_position_sampler.sh <option>"
  echo ""
  echo "available options:"
  echo "    all       running both cropping and sampling"
  echo "    crop      generating cropped images"
  echo "    sample    sampling skewering positions"
  echo "    view      checking saved skewering positions"
  exit
}

if [ -z $1 ]; then
  print_usage
fi

if [[ $1 == 'all' || $1 == 'crop' || $1 == 'sample' ]]; then
  echo 'option:' $1
else
  print_usage
fi


python_version=`cat ./config/python_version`

cd ./scripts

$python_version ./skewering_position_sampler.py $1

cd -

