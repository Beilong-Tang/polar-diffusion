#!/bin/bash
# CIFAR-10: train -> generate 50k samples.
# Usage: bash shell/cifar10.sh [stage] [stop_stage]
set -e

stage=${1:-0}
stop_stage=${2:-1}

DATA_ROOT=datasets          # CIFAR-10 is downloaded here if missing
CONFIG=configs/cifar10.json
EXP_NAME=cifar10_polar
NUM_GPUS=4
EPOCH=2040                    # checkpoint to sample from
SAMPLE_METHOD=polar           # polar | ddpm
CKPT=chkpts/${EXP_NAME}/${EXP_NAME}_${EPOCH}.pt
SAVE_DIR=output/eval/${EXP_NAME}/${EXP_NAME}_${EPOCH}_${SAMPLE_METHOD}

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "[stage 0] training"
    python scripts/cifar10/train.py --config-path $CONFIG --exp-name $EXP_NAME --root $DATA_ROOT \
        --distributed --rigid-launch --num-gpus $NUM_GPUS --resume
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "[stage 1] generating samples"
    python scripts/cifar10/generate.py --config-path $CONFIG --chkpt-path $CKPT --root $DATA_ROOT \
        --sample-method $SAMPLE_METHOD --num-gpus $NUM_GPUS --total-size 50000 --save-dir $SAVE_DIR
fi
