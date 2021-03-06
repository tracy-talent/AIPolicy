#!/bin/bash
# $1: dataset, $2: encoder_type[cls, rbert, entity, entity_context], $3: GPU id
dropout_rates=(0.4 0.5)
python_command="
python train_supervised_bert.py \
    --pretrain_path /home/mist/NLP/corpus/transformers/bert-large-uncased-wwm \
    --language en \
    --bert_name bert \
    --encoder_type $2 \
    --dataset $1 \
    --compress_seq \
    --use_attention4context \
    --adv none \
    --loss ce \
    --batch_size 32 \
    --lr 2e-5 \
    --bert_lr 2e-5 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --metric micro_f1 \
    --optimizer adam 
"

if [ $1 == semeval ]
then
    maxlen=128
    negid=1
    ep=10
elif [ $1 == kbp37 ]
then
    maxlen=256
    negid=10
    ep=5
fi

for dpr in ${dropout_rates[*]}
do  
    echo "Run dataset $1: dpr=$dpr"
    PYTHONIOENCODING=utf8 \
    CUDA_VISIBLE_DEVICES=$3 \
    $python_command \
    --max_length $maxlen \
    --neg_classes \[$negid\] \
    --max_epoch $ep \
    --dropout_rate $dpr
done
