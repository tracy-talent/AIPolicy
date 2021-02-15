#!/bin/bash
# $1: dataset, $2: GPU id
dropout_rates=($3)
python_command="
python train_bert_mtl_span_attr_boundary.py \
    --pretrain_path /home/mist/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --model_type ple \
    --dataset $1 \
    --compress_seq \
    --tagscheme bmoes \
    --bert_name bert \
    --span_use_lstm \
    --span_use_crf \
    --attr_use_lstm \
    --batch_size 32 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --optimizer adam \
    --loss ce \
    --adv fgm \
    --metric micro_f1
"

if [ $1 == weibo -o $1 == resume ]
then
    maxlen=200
    maxep=10
else
    maxlen=256
    maxep=5
fi

for dpr in ${dropout_rates[*]}
do  
echo "Run dataset $1: dpr=$dpr"
CUDA_VISIBLE_DEVICES=$2 \
$python_command \
--max_length $maxlen \
--max_epoch $maxep \
--dropout_rate $dpr
done
