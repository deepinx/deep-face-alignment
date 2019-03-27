#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

NETWORK='sat'
MODELDIR='./model/model-sat2d3-cab-3d/2'
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
PRETRAINED="./model/model-sat2d3-cab-3d/1/model,76"
LOGFILE="$MODELDIR/log"

CUDA_VISIBLE_DEVICES='0' python -u train.py --network $NETWORK --prefix "$PREFIX" --pretrained $PRETRAINED --per-batch-size 16 --lr 4e-6 --lr-epoch-step '5,10' > "$LOGFILE" 2>&1 &


# NETWORK='sat'
# MODELDIR='./model/model-sat2d3-cab/3'
# mkdir -p "$MODELDIR"
# PREFIX="$MODELDIR/model"
# PRETRAINED="./model/model-sat2d3-cab/2/model,0"
# LOGFILE="$MODELDIR/log"

# CUDA_VISIBLE_DEVICES='0' python -u train.py --network $NETWORK --prefix "$PREFIX" --pretrained $PRETRAINED --per-batch-size 16 --lr 1e-6 --lr-epoch-step '10' > "$LOGFILE" 2>&1 &



# NETWORK='hourglass'
# MODELDIR='./model/model-hg2d3-cab-l2/3'
# mkdir -p "$MODELDIR"
# PREFIX="$MODELDIR/model"
# PRETRAINED="./model/model-hg2d3-cab-l2/2/model,79"
# LOGFILE="$MODELDIR/log"

# CUDA_VISIBLE_DEVICES='0' python -u train.py --network $NETWORK --prefix "$PREFIX" --pretrained $PRETRAINED --per-batch-size 24 --lr 1e-6 --lr-epoch-step '20,35,45' > "$LOGFILE" 2>&1 &


# NETWORK='hourglass'
# MODELDIR='./model/model-hg2d4-resnet/1'
# mkdir -p "$MODELDIR"
# PREFIX="$MODELDIR/model"
# PRETRAINED="./model/model-hg2d3-hpm/1/model,100"
# LOGFILE="$MODELDIR/log"

# CUDA_VISIBLE_DEVICES='0' python -u train.py --network $NETWORK --prefix "$PREFIX" --per-batch-size 34 --lr 1e-4 --lr-epoch-step '20,35,45' > "$LOGFILE" 2>&1 &