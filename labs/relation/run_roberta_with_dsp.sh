#!/bin/bash
# $1: dataset, $2: encoder_type[dsp, entity_dsp, entity_context_dsp], $3: GPU id
dropout_rates=(0.1 0.2 0.3 0.4 0.5)
python_command="
python train_supervised_bert_with_dsp.py \
    --pretrain_path /home/mist/NLP/corpus/transformers/roberta-large-cased \
    --language en \
    --bert_name roberta \
    --encoder_type $2 \
    --dataset $1 \
    --compress_seq \
    --dsp_preprocessed \
    --use_attention4context \
    --use_attention4dsp \
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
    --dsp_tool stanza \
    --optimizer adam 
"

if [ $1 == semeval ]
then
    maxlen=128
    max_dsp_path_len=15
    negid=1
elif [ $1 == kbp37 ]
then
    maxlen=200
    max_dsp_path_len=22
    negid=10
fi

for dpr in ${dropout_rates[*]}
do  
    echo "Run dataset $1: dpr=$dpr"
    CUDA_VISIBLE_DEVICES=$3 \
    $python_command \
    --max_length $maxlen \
    --max_dsp_path_length $max_dsp_path_len \
    --neg_classes \[$negid\] \
    --dropout_rate $dpr
done
