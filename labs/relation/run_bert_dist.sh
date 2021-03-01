#!/bin/bash
# $1: dataset, $2: GPU id
dropout_rates=(0.1 0.2 0.3 0.4 0.5)
python_command="
python train_supervised_bert_dist.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/google-bert-large-uncased-wwm \
    --language en \
    --bert_name bert \
    --encoder_type entity_dist_pcnn \
    --dataset $1 \
    --compress_seq \
    --adv none \
    --loss ce \
    --position_size 50 \
    --batch_size 16 \
    --lr 2e-5 \
    --bert_lr 2e-5 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_epoch 10 \
    --metric micro_f1 \
    --optimizer adam 
"

if [ $1 == semeval ]
then
    maxlen=128
    negid=1
elif [ $1 == kbp37 ]
then
    maxlen=200
    negid=10
fi

for dpr in ${dropout_rates[*]}
do  
    echo "Run dataset $1: dpr=$dpr"
    CUDA_VISIBLE_DEVICES=$2 \
    $python_command \
    --max_length $maxlen \
    --neg_classes \[$negid\] \
    --dropout_rate $dpr
done
