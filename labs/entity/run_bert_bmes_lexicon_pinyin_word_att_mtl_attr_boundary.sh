#!/bin/bash
# $1: dataset, $2: word2vec_file, $3: pinyin2vec_file, $4: GPU id
python_command="
python train_bert_bmes_lexicon_pinyin_att_mtl_attr_boundary.py \
    --pretrain_path /home/ghost/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --word2pinyin_file /home/ghost/NLP/corpus/pinyin/word2pinyin_num5.txt \
    --pinyin_embedding_type word_att_add \
    --model_type ple \
    --compress_seq \
    --group_num 3 \
    --dataset $1 \
    --tagscheme bmoes \
    --bert_name bert \
    --experts_layers 2 \
    --experts_num 1 \
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

if [ $1 == resume]
then
    seed=12345
    bz=16
    pact=elu
    pdpr=0
    ld=1
    dropout_rates=(0.1)
    lexicon_window_sizes=(13)
elif [ $1 == msra ]
then
    seed=12345
    bz=64
    pact=gelu
    pdpr=0.1
    ld=0.55
    dropout_rates=(0.2)
    lexicon_window_sizes=(9)
elif [ $1 == policy ]
then
    seed=12345
    bz=8
    pact=gelu
    pdpr=0.2
    ld=1.0
    dropout_rates=(0.3)
    lexicon_window_sizes=(3)
fi

if [ $1 == weibo -o $1 == resume ]
then
    maxlen=200
    maxep=15
elif [ $1 == policy ]
then
    maxlen=256
    maxep=20
else
    maxlen=256
    maxep=5
fi

if [ $2 == sgns ]
then
    lexicon2vec=sgns_merge_word.1293k.300d.bin
    pinyin_dim=300
else
    lexicon2vec=ctb.704k.50d.bin
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
    --word2vec_file /home/ghost/NLP/corpus/embedding/chinese/lexicon/$lexicon2vec \
    --pinyin2vec_file /home/ghost/NLP/corpus/pinyin/$pinyin2vec \
    --random_seed $seed \
    --max_length $maxlen \
    --max_epoch $maxep \
    --dropout_rate $dpr \
    --lexicon_window_size $lws \
    --batch_size $bz \
    --pactivation $pact \
    --ple_dropout $pdpr \
    --lr_decay $ld
    done
done
