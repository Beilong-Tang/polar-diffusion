#!/bin/bash
# ImageNet-64 (unconditional): radius statistics -> train -> generate 10k samples.
# Usage: bash shell/imagenet64.sh [stage] [stop_stage]
set -e

stage=${1:-0}
stop_stage=${2:-2}

# Training data: a folder of 64x64 images, a .npy of uint8 images [N, 64, 64, 3],
# or a WebDataset shard pattern such as "/data/imagenet64-{0001..1282}.tar".
DATA_DIR=/path/to/imagenet64.npy
RADIUS_STATS=configs/imagenet64_radius.json
LOGDIR=logs/imagenet64_polar
NUM_GPUS=1
STEP=200000                       # checkpoint to sample from
SAMPLE_METHOD=polar               # polar | ddpm
POLAR_BETA_THRESHOLD=-1           # > 0: DDPM steps where beta_t exceeds it (hybrid sampler)
OUTDIR=$LOGDIR/eval_${SAMPLE_METHOD}_${STEP}

MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --radius_stats $RADIUS_STATS"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 192 --total_step 200000 --num_workers 8"  # batch size per GPU

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    if [ ! -f $RADIUS_STATS ]; then
        echo "[stage 0] radius statistics"
        python scripts/imagenet64/radius_stats.py --data $DATA_DIR --out $RADIUS_STATS
    fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "[stage 1] training"
    torchrun --standalone --nproc_per_node=$NUM_GPUS scripts/imagenet64/train.py \
        --data_dir "$DATA_DIR" --logdir $LOGDIR $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "[stage 2] generating samples"
    torchrun --standalone --nproc_per_node=$NUM_GPUS scripts/imagenet64/sample.py \
        $MODEL_FLAGS $DIFFUSION_FLAGS --model_path $LOGDIR/ema_0.9999_${STEP}.pt --outdir $OUTDIR \
        --num_samples 10000 --batch_size 64 --sample_method $SAMPLE_METHOD --polar_beta_threshold $POLAR_BETA_THRESHOLD
fi
