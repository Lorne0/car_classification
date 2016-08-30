#!/usr/bin/env sh
/root/caffe/build/tools/caffe train \
-solver finetune_solver.prototxt \
-weights googlenet_finetune_web_car_iter_15000.caffemodel \
-gpu=3 \
