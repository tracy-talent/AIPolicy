#!/bin/bash
# $1: dataset, $2: encoder type[entity_dist, entity_dist_pcnn], $3: GPU id
dropout_rates=(0.1 0.2 0.3 0.4 0.5)
#dropout_rates=(0.1)
python_command="
python train_supervised_bert_dist.py \
    --pretrain_path /home/mist/NLP/corpus/transformers/roberta-large-cased \
    --language en \
    --bert_name roberta \
    --encoder_type $2 \
    --dataset $1 \
    --compress_seq \
    --adv none \
    --loss ce \
    --position_size 10 \
    --batch_size 32 \
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
    CUDA_VISIBLE_DEVICES=$3 \
    $python_command \
    --max_length $maxlen \
    --neg_classes \[$negid\] \
    --dropout_rate $dpr
done