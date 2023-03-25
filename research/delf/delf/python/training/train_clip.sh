#!/bin/bash
# Created Time: Wed 01 Mar 2023 03:59:36 PM PST
# Author: Taihong Xiao <xiaotaihong@126.com>

# export PYTHONPATH="${PYTHONPATH}:/home/taihong/work/retrieval/models/research/delf"
# export PYTHONPATH="${PYTHONPATH}:/home/taihong/work/retrieval/models/research/delf"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/taihong/.local/anaconda3/envs/clip-tf2/lib

# without using clip weight
CUDA_VISIBLE_DEVICES=5 python train_clip.py  \
    --train_file_pattern=/tmp3/taihong/data/GLDv2-clean-tfrecord/train* \
    --validation_file_pattern=/tmp3/taihong/data/GLDv2-clean-tfrecord/validation*  \
    --imagenet_checkpoint=/tmp3/taihong/data/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    --text_weight_ckpt=/home/taihong/work/retrieval/CLIP-tf2/models/linear_weights/text_weight.npy \
    --dataset_version=gld_v2_clean \
    --logdir=gldv2_training \
    --delg_global_features \
    --use_clip_backbone=False \
    --debug

# # using pretrained clip weight
# CUDA_VISIBLE_DEVICES=5 python train_clip.py  \
#     --train_file_pattern=/tmp3/taihong/data/GLDv2-clean-tfrecord/train* \
#     --validation_file_pattern=/tmp3/taihong/data/GLDv2-clean-tfrecord/validation*  \
#     --imagenet_checkpoint=/home/taihong/work/retrieval/clip_resnet/models/CLIP_image_RN50/ckpt \
#     --text_weight_ckpt=/home/taihong/work/retrieval/CLIP-tf2/models/linear_weights/text_weight.npy \
#     --dataset_version=gld_v2_clean \
#     --logdir=gldv2_training \
#     --delg_global_features \
#     --debug

