#!/bin/bash
if [ $1 == resume ];then
batch_size=32
max_len=200
max_epoch=10
elif [ $1 == weibo ];then
batch_size=32
max_len=200
max_epoch=20
elif [ $1 == msra ]; then
batch_size=32
max_len=250
max_epoch=8
elif [ $1 == ontonotes4 ]; then
batch_size=32
max_len=250
max_epoch=6
fi

echo "$1: bz=$batch_size, max_len=$max_len, epochs=$max_epoch"
CUDA_VISIBLE_DEVICES=$2 \
python train_bert_crf.py \
    --pretrain_path /home/ghost/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --bert_name bert \
    --dataset $1 \
    --tagscheme bmoes \
    --compress_seq \
    --use_lstm \
    --use_crf \
    --batch_size $batch_size \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length $max_len \
    --max_epoch $max_epoch \
    --optimizer adam \
    --loss ce \
    --adv fgm \
    --dice_alpha 0.6 \
    --metric micro_f1 \
    --random_seed 2
