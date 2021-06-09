#!/bin/bash
# $1: dataset, $2: GPU id

python_command="
python train_bert_bmes_lexicon_pinyin_att_mtl_span_attr_boundary.py \
    --pretrain_path /home/ghost/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --word2vec_file /home/ghost/NLP/corpus/embedding/chinese/lexicon/ctb.704k.50d.bin \
    --pinyin2vec_file /home/ghost/NLP/corpus/pinyin/word2vec/word2vec_num5.1409.50d.vec \
    --compress_seq \
    --pinyin_embedding_type word_att_add \
    --group_num 3 \
    --model_type ple \
    --experts_layers 2 \
    --experts_num 1 \
    --use_ff 0 \
    --dataset $1 \
    --tagscheme bmoes \
    --bert_name bert \
    --span_use_lstm \
    --span_use_crf \
    --attr_use_lstm \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_pinyin_char_length 7 \
    --pinyin_char_embedding_size 50 \
    --optimizer adam \
    --loss ce \
    --adv none \
    --metric micro_f1 \
"

if [ $1 == weibo -o $1 == weibo.ne -o $1 == weibo.nm ]
then
    seed=2
    bz=64
    pact=gelu
    pdpr=0.6
    ld=1.0
    dropout_rates=(0.4)
    lexicon_window_sizes=(2 3 4 5 6 7)
elif [ $1 == resume ]
then
    seed=12345
    bz=8
    pact=elu
    pdpr=0
    ld=1.0
    dropout_rates=(0.1)
    #lexicon_window_sizes=(13 12 11 10 9 8 7 6 5 4 3 2)
    lexicon_window_sizes=(12)
elif [ $1 == ontonotes4 ]
then
    seed=12345
    bz=32
    pact=gelu
    pdpr=0.1
    ld=0.42
    dropout_rates=(0.3)
    lexicon_window_sizes=(2 3 4 5 6 7 8)
elif [ $1 == msra ]
then
    seed=12345
    bz=64
    pact=gelu
    pdpr=0.1
    ld=0.55
    dropout_rates=(0.2)
    lexicon_window_sizes=(2 3 4 5 6 7 8 9)
elif [ $1 == policy ]
then
    seed=12345
    bz=8
    pact=gelu
	#lws=3
	#dpr=0.3
	pdpr=0.2
	ld=1.0
    #pdprs=(0.2)
	#lds=(1.0)
    dropout_rates=(0.3)
    lexicon_window_sizes=(2 4)
fi

if [ $1 == weibo -o $1 == weibo.ne -o $1 == weibo.nm ]
then
    maxlen=200
    maxep=20
elif [ $1 == resume ]
then
    maxlen=200
    maxep=10
elif [ $1 == policy ]
then
    maxlen=256
    maxep=20
else 
    maxlen=256
    maxep=5
fi

for lws in ${lexicon_window_sizes[*]}
#for ld in ${lds[*]}
do
    for dpr in ${dropout_rates[*]}
	#do
	#for pdpr in ${pdprs[*]}
    do  
    echo "Run dataset $1: batch_size=$bz, dropout_rate=$dpr, lexicon_window_size=$lws"
    CUDA_VISIBLE_DEVICES=$2 \
    $python_command \
    --max_length $maxlen \
    --max_epoch $maxep \
    --dropout_rate $dpr \
    --lexicon_window_size $lws \
    --batch_size $bz \
    --ple_dropout $pdpr \
    --pactivation $pact \
    --lr_decay $ld \
    --random_seed $seed \
    #--only_test 
    done
	#done
done

#datestr=`date +%Y-%m-%d`
#if [ ! -x /data/labs/$datestr ]; then
#    mkdir /data/labs/entity_$datestr
#fi
#
#cp -r ~/github/AIPolicy/output/entity/logs/* /data/labs/entity_$datestr
#sh ~/shutdown.sh
