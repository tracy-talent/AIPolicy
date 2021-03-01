#!/bin/bash
# $1: dataset, $2: encoder_tyep[entity, entity_context], GPU id
dropout_rates=(0.1 0.2 0.3 0.4 0.5)
python_command="
python train_supervised_xlnet.py \
    --pretrain_path /home/mist/NLP/corpus/transformers/xlnet-large-cased \
    --language en \
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
    --max_epoch 10 \
    --metric micro_f1 \
    --optimizer adam 
"

if [ $1 == semeval ]
then
    maxlen=150
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
    --neg_classes \[$negid\] \
    --max_length $maxlen \
    --dropout_rate $dpr
done
