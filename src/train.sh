#!/usr/bin/env bash

#arch: resnet_18 or fpn_resnet_18

python train.py \
  --root-dir '../' \
  --saved_fn 'rtm3d_resnet_18' \
  --arch 'resnet_18' \
  --batch_size 16 \
  --num_workers 4 \
  --gpu_idx 0 \
  --hflip_prob 0.5 \
  --use_left_cam_prob 1.
