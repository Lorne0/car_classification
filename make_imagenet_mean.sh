#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/root/car
DATA=/root/car
TOOLS=/root/caffe/build/tools

$TOOLS/compute_image_mean $EXAMPLE/compcars_test_lmdb \
  $DATA/test_mean.binaryproto

echo "Done."
