#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice


NETWORK='satnet'
MODELDIR='./model/model-sat2d3-cab-3d/2'
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
PRETRAINED="./model/model-sat2d3-cab-3d/1/model,76"
LOGFILE="$MODELDIR/log"

CUDA_VISIBLE_DEVICES='0' python -u train.py --network $NETWORK --prefix "$PREFIX" --pretrained $PRETRAINED --per-batch-size 16 --lr 2e-6 --lr-epoch-step '5,10' > "$LOGFILE" 2>&1 &



# NETWORK='hourglass'
# MODELDIR='./model/model-hg2d3-cab-3d/3'
# mkdir -p "$MODELDIR"
# PREFIX="$MODELDIR/model"
# PRETRAINED="./model/model-hg2d3-cab-3d/2/model,8"
# LOGFILE="$MODELDIR/log"

# CUDA_VISIBLE_DEVICES='0' python -u train.py --network $NETWORK --prefix "$PREFIX" --pretrained $PRETRAINED --per-batch-size 24 --lr 1e-6 --lr-epoch-step '20,35,45' > "$LOGFILE" 2>&1 &