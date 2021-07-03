#!/bin/bash
# $1: dataset, $2: GPU id
if [ $1 == resume ]; then
  # RESUME
  batch_size=64
  max_epoch=20
  dropout_rates=(0.1)
  lexicon_window_sizes=(11) #  13 12 10 9 8 7 6 5 4 3 2
  random_seeds=12345
  ple_dropouts=(0.0)
  pactivation=elu
  decay_rate=1.0
  maxlen=200
elif [ $1 == weibo ]; then
  #WEIBO
  batch_size=32
  max_epoch=10
  dropout_rates=(0.5)
  lexicon_window_sizes=(5)
  random_seeds=(2)
  ple_dropouts=(0.1)
  pactivation=gelu
  decay_rate=1.0
  maxlen=200
elif [ $1 == msra ];then
  # MSRA
  batch_size=32
  max_epoch=5
  dropout_rates=0.2
  lexicon_window_sizes=(9)
  random_seeds=12345
  ple_dropouts=0.1
  pactivation=gelu
  decay_rate=(0.4)
  maxlen=256
elif [ $1 == ontonotes4 ];then
  # ONTONOTES4
  batch_size=32
  max_epoch=5
  dropout_rates=(0.5)
  lexicon_window_sizes=(5)  #  2 3 4 6 7 8
  random_seeds=12345
  ple_dropouts=0.1
  pactivation=gelu
  decay_rate=(0.56)
  maxlen=256
else
  batch_size=32
  max_epoch=10
  dropout_rates=(0.5)
  lexicon_window_sizes=(5)
  random_seeds=(6)
  ple_dropouts=(0.1)
  pactivation=gelu
  decay_rate=1.0
  maxlen=200
fi


experts_layers=2
experts_num=1
span_loss_weight=-1 # -1表示不调整span_loss，还是1/3
use_ff=0  # 是否使用feedforward网络

python_command="
python train_bert_mtl_span_attr_boundary.py \
    --pretrain_path /root/qmc/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --model_type ple \
    --compress_seq \
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
    --optimizer adam \
    --loss ce \
    --adv fgm \
    --metric micro_f1
"


for seed in ${random_seeds[*]}
do
  for lr_decay in ${decay_rate[*]}
  do
    for dpr in ${dropout_rates[*]}
      do
        for pdpr in ${ple_dropouts[*]}
        do
          echo "Run dataset $1: bz=$batch_size, dpr=$dpr, pdpr=$pdpr, seed=$seed"
          CUDA_VISIBLE_DEVICES=$2 \
            $python_command \
            --max_length $maxlen \
            --max_epoch $max_epoch \
            --dropout_rate $dpr \
            --batch_size $batch_size \
            --experts_layers $experts_layers \
            --experts_num $experts_num \
            --random_seed $seed \
            --ple_dropout $pdpr \
            --pactivation $pactivation \
            --use_ff $use_ff \
            --lr_decay $lr_decay
        done
      done
  done
done
