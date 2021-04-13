#!/bin/bash
# $1: dataset, $2: word2vec_file, $3: GPU id
batch_size=32
dropout_rates=(0.1)
max_epochs=20
lexicon_window_sizes=(13)
python_command="
python train_bert_bmes_lexicon_att_mtl_span_attr_boundary.py \
    --pretrain_path /root/qmc/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --embedding_fusion_type att_add \
    --group_num 3 \
    --model_type ple \
    --dataset $1 \
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
    --compress_seq \
    --metric micro_f1
"

if [ $1 == weibo -o $1 == resume ]
then
    maxlen=200
    maxep=15
else
    maxlen=256
    maxep=5
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
      for ((integer=1; integer<=1; integer++))
        do
        echo "Run dataset $1: dpr=$dpr, wz=$lws"
        CUDA_VISIBLE_DEVICES=$3 \
        $python_command \
        --word2vec_file /root/qmc/NLP/corpus/embedding/chinese/lexicon/$lexicon2vec \
        --max_length $maxlen \
        --max_epoch $max_epochs \
        --batch_size $batch_size \
        --dropout_rate $dpr \
        --lexicon_window_size $lws \
        --only_test \
        --ckpt /root/qmc/github/AIPolicy/output/entity/ckpt_3.28_wopinyin/$1_bmoes/bmes3_lexicon_ctb_window13_mtl_span_attr_boundary_ple_bert_relu_crf1e-03_bert3e-05_spanlstm_attrlstm_spancrf_ce_fgm_dpr0.1_micro_f1_1.pth.tar
        done
    done
done