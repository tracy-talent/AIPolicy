#!/bin/bash
# $1: dataset, $2: word2vec_file, $3: pinyin2vec_file, $4: GPU id
dropout_rates=(0.5)
lexicon_window_sizes=(4)
python_command="
python train_bert_bmes_lexicon_pinyin_freqasweight_mtl_span_attr_boundary.py \
    --pretrain_path /root/qmc/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --word2pinyin_file /root/qmc/NLP/corpus/pinyin/word2pinyin_num5.txt \
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
    --max_pinyin_char_length 7 \
    --pinyin_char_embedding_size 50 \
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
    maxep=20
fi

if [ $2 == sgns ]
then
    lexicon2vec=sgns_merge_word.1293k.300d.bin
    pinyin_dim=300
else
    lexicon2vec=ctbword_gigachar_mix.710k.50d.bin
    pinyin_dim=50
fi

if [ $3 == glove ]
then
    pinyin2vec=glove/glove_num5.1409.${pinyin_dim}d.vec
else
    pinyin2vec=word2vec/word2vec_num5.1409.${pinyin_dim}d.vec
fi

for lws in ${lexicon_window_sizes[*]}
do
    for dpr in ${dropout_rates[*]}
    do
        for((integer=1; integer<=2; integer++))
        do
        echo "Run dataset $1: dpr=$dpr, wz=$lws"
        PYTHONIOENCODING=utf8 \
        CUDA_VISIBLE_DEVICES=$4 \
        $python_command \
        --word2vec_file /root/qmc/NLP/corpus/embedding/chinese/lexicon/$lexicon2vec \
        --pinyin2vec_file /root/qmc/NLP/corpus/pinyin/$pinyin2vec \
        --max_length $maxlen \
        --max_epoch $maxep \
        --dropout_rate $dpr \
        --adv fgm \
        --lexicon_window_size $lws
        done
    done
done
