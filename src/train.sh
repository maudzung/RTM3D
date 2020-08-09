#!/usr/bin/env bash
python train.py \
  --root-dir '../' \
  --saved_fn 'rtm3d_resnet_18' \
  --arch 'resnet_18' \
  --batch_size 16 \
  --num_workers 4 \
  --gpu_idx 0
