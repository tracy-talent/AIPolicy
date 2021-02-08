#!/bin/bash
# $1: dataset, $2: word2vec_file, $3: GPU id
dropout_rates=(0.1 0.2 0.3 0.4 0.5)
lexicon_window_sizes=(4)
python_command="
python train_bert_bmes_lexicon_att_mtl_span_attr_boundary.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --embedding_fusion_type att_add \
    --group_num 3 \
    --model_type ple \
    --dataset $1 \
    --compress_seq \
    --tagscheme bmoes \
    --bert_name bert \
    --span_use_lstm \
    --span_use_crf \
    --attr_use_lstm \
    --batch_size 16 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_epoch 5 \
    --optimizer adam \
    --loss ce \
    --adv fgm \
    --metric micro_f1
"

if [ $1 == weibo -o $1 == resume ]
then
    maxlen=200
else
    maxlen=256
fi

if [ $2 == sgns ]
then
    lexicon2vec=sgns_merge_word.1293k.300d.bin
else
    lexicon2vec=ctbword_gigachar_mix.710k.50d.bin
fi

for lws in ${lexicon_window_sizes[*]}
do
    for dpr in ${dropout_rates[*]}
    do  
    echo "Run dataset $1: dpr=$dpr, wz=$lws"
    CUDA_VISIBLE_DEVICES=$3 \
    $python_command \
    --word2vec_file /home/liujian/NLP/corpus/embedding/chinese/lexicon/$lexicon2vec \
    --max_length $maxlen \
    --dropout_rate $dpr \
    --lexicon_window_size $lws
    done
done
