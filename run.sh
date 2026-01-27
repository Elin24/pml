#!/bin/bash

T=`date +%m%d%H%M%S`
#T='no_mask'

mkdir exp
mkdir exp/$T
mkdir exp/$T/code
cp -r datasets exp/$T/code/datasets
cp -r models exp/$T/code/models
cp -r losses exp/$T/code/losses
cp ./*.py exp/$T/code/
cp run.sh exp/$T/code

mkdir exp/$T/train.log

datadir=/home/visal/wlin38/crowd/data/QNRF_PNG_SE_1536/

# Train
python main.py \
     --data-path $datadir \
     --batch-size 16 --device cuda:1 --tag $T 2>&1 | tee exp/$T/train.log/running.log

# Eval
# python main.py \
#      --data-path $datadir \
#      --eval --resume model_ckpt.pth
#      --batch-size 1 --device cuda:1 --tag $T 2>&1 | tee exp/$T/train.log/running.log
