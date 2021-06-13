#!/bin/bash
# $1: dataset, $2: word2vec_file, $3: pinyin2vec_file, $4: GPU id
if [ $1 == resume ]; then
# RESUME
batch_size=64
max_epoch=20
dropout_rates=(0.1)
lexicon_window_sizes=(9 8 7 6 5 4 3 2)
random_seeds=12345
ple_dropouts=(0.0)
pactivation=elu
decay_rate=1.0
elif [ $1 == weibo ]; then
#WEIBO
batch_size=32
max_epoch=10
dropout_rates=(0.5)
lexicon_window_sizes=(5)
random_seeds=12345
ple_dropouts=(0.1)
pactivation=gelu
decay_rate=1.0
elif [ $1 == msra ];then
# MSRA
batch_size=32
max_epoch=5
dropout_rates=0.2
lexicon_window_sizes=(11)
random_seeds=12345
ple_dropouts=0.1
pactivation=gelu
decay_rate=(0.38 0.37 0.36)
#decay_rate=(0.41 0.39 0.45 0.35 0.6)
elif [ $1 == ontonotes4 ];then
# ONTONOTES4
batch_size=32
max_epoch=5
dropout_rates=(0.5)
lexicon_window_sizes=(5)
random_seeds=12345
ple_dropouts=0.1
pactivation=gelu
decay_rate=(0.56 0.505 0.555)
fi

#random_seeds=34
#ple_dropouts=0.6     # PLE里面使用dropout rate
#pactivation=gelu  # PLE里面使用的激活函数
#decay_rate=1.0  # 针对resume和msra使用的学习率衰减
experts_layers=2
experts_num=1
span_loss_weight=-1 # -1表示不调整span_loss，还是1/3
use_ff=0  # 是否使用feedforward网络

duplicate_run_rounds=1
python_command="
python train_bert_bmes_lexicon_pinyin_att_mtl_span_attr_boundary.py \
    --pretrain_path /root/qmc/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --word2vec_file /root/qmc/NLP/corpus/embedding/chinese/lexicon/ctb.704k.50d.bin \
    --pinyin2vec_file /root/qmc/NLP/corpus/pinyin/word2vec/word2vec_num5.1409.50d.vec \
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
    --compress_seq
"

if [ $1 == weibo -o $1 == resume -o $1 == none ]
then
    maxlen=200
else
    maxlen=256
fi


for epn in ${experts_num[*]}
do
  for seed in ${random_seeds[*]}
  do
    for lr_decay in ${decay_rate[*]}
    do
      for lwz in ${lexicon_window_sizes[*]}
      do
        for dpr in ${dropout_rates[*]}
        do
          for pdpr in ${ple_dropouts[*]}
          do
            echo "Run dataset $1: bz=$batch_size, dpr=$dpr, pdpr=$pdpr, wz=$lwz, seed=$seed"
            CUDA_VISIBLE_DEVICES=$2 \
            $python_command \
            --max_length $maxlen \
            --max_epoch $max_epoch \
            --dropout_rate $dpr \
            --lexicon_window_size $lwz \
            --batch_size $batch_size \
            --experts_layers $experts_layers \
            --experts_num $epn \
            --random_seed $seed \
            --ple_dropout $pdpr \
            --span_loss_weight $span_loss_weight \
            --pactivation $pactivation \
            --use_ff $use_ff \
            --lr_decay $lr_decay \
  #          --train_file /root/qmc/github/AIPolicy/input/benchmark/entity/weibo/train.ne.bmoes \
  #          --val_file /root/qmc/github/AIPolicy/input/benchmark/entity/weibo/dev.ne.bmoes \
  #          --test_file /root/qmc/github/AIPolicy/input/benchmark/entity/weibo/test.ne.bmoes \
  #          --attr2id_file /root/qmc/github/AIPolicy/input/benchmark/entity/weibo/attr2id.ne.bmoes \
  #          --span2id_file /root/qmc/github/AIPolicy/input/benchmark/entity/weibo/span2id.bmoes \
          done
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
