#!/usr/bin/env sh
/root/caffe/build/tools/caffe train \
-solver solver_googlenet.prototxt \
-weights imagenet_googlenet.caffemodel \
-gpu=3 \
