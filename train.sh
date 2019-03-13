#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

# NETWORK='sdu'
# MODELDIR='./model/model-sdu-3d/3'
# mkdir -p "$MODELDIR"
# PREFIX="$MODELDIR/model"
# PRETRAINED="./model/model-sdu-3d/2/model,35"
# LOGFILE="$MODELDIR/log"

# CUDA_VISIBLE_DEVICES='0' python -u train.py --network $NETWORK --prefix "$PREFIX" --pretrained $PRETRAINED --per-batch-size 16 --lr 0.00002 --lr-step '16000,24000,30000' > "$LOGFILE" 2>&1 &


# NETWORK='sdu'
# MODELDIR='./model/model-sat2-cab3/2'
# mkdir -p "$MODELDIR"
# PREFIX="$MODELDIR/model"
# PRETRAINED="./model/model-sat2-cab3/1/model,150"
# LOGFILE="$MODELDIR/log"

# CUDA_VISIBLE_DEVICES='0' python -u train.py --network $NETWORK --prefix "$PREFIX" --pretrained $PRETRAINED --per-batch-size 16 --lr 1e-6 --lr-step '20000' > "$LOGFILE" 2>&1 &



NETWORK='hourglass'
MODELDIR='./model/model-hg2-hpm3/3'
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
PRETRAINED="./model/model-hg2-hpm3/2/model,4"
LOGFILE="$MODELDIR/log"

CUDA_VISIBLE_DEVICES='0' python -u train.py --network $NETWORK --prefix "$PREFIX" --pretrained $PRETRAINED --per-batch-size 25 --lr 1e-6 --lr-step '16000,24000,30000' > "$LOGFILE" 2>&1 &