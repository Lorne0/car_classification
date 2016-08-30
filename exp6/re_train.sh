#!/usr/bin/env sh
/root/caffe/build/tools/caffe train \
-solver solver_googlenet.prototxt \
-snapshot mix_finetune/googlenet_finetune_mix_car_iter_1000.solverstate \
-gpu=3 \
