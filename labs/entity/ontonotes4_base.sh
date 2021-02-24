#!/bin/bash
# $1: dataset, $2: word2vec_file, $3: pinyin2vec_file, $4: GPU id
dropout_rates=(0.3)
lexicon_window_sizes=(8 4 7 6 5)
python_command="
python train_base_bmes_lexicon_pinyin_att_mtl_span_attr_boundary.py \
    --token2vec_file /home/liujian/NLP/corpus/embedding/chinese/lexicon/gigaword_chn.all.a2b.uni.11k.50d.bin \
    --word2pinyin_file /home/liujian/NLP/corpus/pinyin/word2pinyin_num5.txt \
    --pinyin_embedding_type word_att_cat \
    --group_num 3 \
    --model_type ple \
    --dataset $1 \
    --compress_seq \
    --tagscheme bmoes \
    --bert_name bert \
    --span_use_crf \
    --batch_size 10 \
    --crf_lr 1e-3 \
    --lr 1e-3 \
    --bert_lr 1e-3 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 5 \
    --max_pinyin_char_length 7 \
    --pinyin_char_embedding_size 50 \
    --optimizer adam \
    --loss ce \
    --metric micro_f1
"

if [ $1 == weibo -o $1 == resume ]
then
    maxlen=200
    maxep=150
else
    maxlen=256
    maxep=500
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
    echo "Run dataset $1: dpr=$dpr, wz=$lws"
    CUDA_VISIBLE_DEVICES=$4 \
    $python_command \
    --word2vec_file /home/liujian/NLP/corpus/embedding/chinese/lexicon/$lexicon2vec \
    --pinyin2vec_file /home/liujian/NLP/corpus/pinyin/$pinyin2vec \
    --max_length $maxlen \
    --max_epoch $maxep \
    --dropout_rate $dpr \
    --lexicon_window_size $lws
    done
done
