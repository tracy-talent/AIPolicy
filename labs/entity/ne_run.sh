#!/bin/bash
# $1: dataset, $2: word2vec_file, $3: pinyin2vec_file, $4: GPU id

#WEIBO
batch_size=64
max_epoch=20
dropout_rates=0.4
lexicon_window_sizes=(5)


experts_layers=(2)
experts_num=(1)
#random_seeds=(0 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)
ple_dropout=(0.6)     # PLE里面使用dropout rate
span_loss_weight=-1 # -1表示不调整span_loss，还是1/3
pactivation=gelu  # PLE里面使用的激活函数
use_ff=0  # 是否使用feedforward网络
decay_rate=(1)  # 针对resume和msra使用的学习率衰减

duplicate_run_rounds=1
python_command="
python train_bert_bmes_lexicon_pinyin_att_mtl_span_attr_boundary.py \
    --pretrain_path /root/qmc/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --word2pinyin_file /root/qmc/NLP/corpus/pinyin/word2pinyin_num5.txt \
    --pinyin_embedding_type word_att_add \
    --group_num 3 \
    --model_type ple \
    --dataset $1 \
    --tagscheme bmoes \
    --bert_name bert \
    --span_use_lstm \
    --span_use_crf \
    --attr_use_lstm \
    --batch_size 64 \
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
    --metric micro_f1 \
"

if [ $1 == weibo -o $1 == resume -o $1 == none ]
then
    maxlen=200
    maxep=20
else
    maxlen=256
    maxep=10
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

for epl in ${experts_layers[*]}
do
  for epn in ${experts_num[*]}
  do
    for ((seed=34; seed <= 34; seed++))   # seed in ${random_seed[*]}
    do
      for lr_decay in ${decay_rate[*]}
      do
        for lwz in ${lexicon_window_sizes[*]}
        do
          echo "Run dataset $1: bz=$batch_size, dpr=$dropout_rates, wz=$lwz, seed=$seed"
          CUDA_VISIBLE_DEVICES=$4 \
          $python_command \
          --word2vec_file /root/qmc/NLP/corpus/embedding/chinese/lexicon/$lexicon2vec \
          --pinyin2vec_file /root/qmc/NLP/corpus/pinyin/$pinyin2vec \
          --max_length $maxlen \
          --max_epoch $max_epoch \
          --dropout_rate $dropout_rates \
          --lexicon_window_size $lwz \
          --batch_size $batch_size \
          --experts_layers $epl \
          --experts_num $epn \
          --random_seed $seed \
          --ple_dropout $ple_dropout \
          --span_loss_weight $span_loss_weight \
          --pactivation $pactivation \
          --use_ff $use_ff \
          --lr_decay $lr_decay \
          --train_file /root/qmc/github/AIPolicy/input/benchmark/entity/weibo/train.nm.bmoes \
          --val_file /root/qmc/github/AIPolicy/input/benchmark/entity/weibo/dev.nm.bmoes \
          --test_file /root/qmc/github/AIPolicy/input/benchmark/entity/weibo/test.nm.bmoes \
          --attr2id_file /root/qmc/github/AIPolicy/input/benchmark/entity/weibo/attr2id.nm.bmoes \
          --span2id_file /root/qmc/github/AIPolicy/input/benchmark/entity/weibo/span2id.bmoes \
  #        --only_test \
  #        --ckpt ~/qmc/github/AIPolicy/output/entity/ckpt_3.26/$1_bmoes/bmes3_lexicon_ctb_window5_pinyin_word_att_add_mtl_span_attr_boundary_base_bert_relu_crf1e-03_bert3e-05_spanlstm_attrlstm_spancrf_ce_fgm_dpr0.4_micro_f1_test_0.pth.tar
        done
      done
    done
  done
done
#
#datestr=`date +%Y-%m-%d`
#if [ ! -x /data/labs/$datestr ]; then
#    mkdir /data/labs/entity_$datestr
#fi
#
#cp -r ~/github/AIPolicy/output/entity/logs/* /data/labs/entity_$datestr
#sh ~/shutdown.sh