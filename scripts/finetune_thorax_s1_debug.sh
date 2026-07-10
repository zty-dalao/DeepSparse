#!/bin/bash

set -e

# Single-GPU debug script for thorax Stage-1, 6-view continuation.
# This script finds the latest checkpoint in logs/thorax+6v+s1 and resumes training.

gpu=0
cfg_path=./configs/finetune_s1.yaml
n_view=6
name=thorax+${n_view}v+s1
log_dir=./logs/$name
mkdir -p "$log_dir"

latest_ckpt=$(find "$log_dir" -maxdepth 1 -name 'ep_*.pth' | sort -V | tail -n 1)
if [ -z "$latest_ckpt" ]; then
    echo "[ERROR] No checkpoint found under $log_dir"
    echo "Please run the normal stage-1 script first."
    exit 1
fi

resume_epoch=$(basename "$latest_ckpt" | sed -n 's/^ep_\([0-9]\+\)\.pth$/\1/p')
if [ -z "$resume_epoch" ]; then
    echo "[ERROR] Failed to parse checkpoint epoch from: $latest_ckpt"
    exit 1
fi

target_epoch=$((resume_epoch + 100))

echo "========================================="
echo "Single-GPU debug continue training: $name"
echo "Latest checkpoint: $latest_ckpt"
echo "Resume epoch: $resume_epoch"
echo "Target epoch: $target_epoch"
echo "========================================="

CUDA_VISIBLE_DEVICES=$gpu python code/train.py \
    --name $name \
    --batch_size 2 \
    --epoch $target_epoch \
    --resume $resume_epoch \
    --dst_name thorax \
    --num_views 24 \
    --min_views $n_view \
    --random_views \
    --cfg_path $cfg_path \
    --vq_w 0.1 \
    >> "$log_dir/train_debug.log" 2>&1

if [ $? -eq 0 ]; then
    echo "Training finished successfully. Log saved to $log_dir/train_debug.log"
else
    echo "Training failed. Check $log_dir/train_debug.log for details."
fi
