#!/bin/bash
# Created Time: Wed 01 Mar 2023 03:59:36 PM PST
# Author: Taihong Xiao <xiaotaihong@126.com>

# export PYTHONPATH="${PYTHONPATH}:/home/taihong/work/retrieval/models/research/delf"
# export PYTHONPATH="${PYTHONPATH}:/home/taihong/work/retrieval/models/research/delf"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/taihong/.local/anaconda3/envs/clip-tf2/lib

CUDA_VISIBLE_DEVICES=0 python train.py  \
    --train_file_pattern=/tmp3/taihong/data/GLDv2-clean-tfrecord/train* \
    --validation_file_pattern=/tmp3/taihong/data/GLDv2-clean-tfrecord/validation*  \
    --imagenet_checkpoint=/tmp3/taihong/data/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    --dataset_version=gld_v2_clean  \
    --logdir=gldv2_training/
