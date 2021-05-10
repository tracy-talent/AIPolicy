#!/bin/bash
# $1: dataset, $2: GPU id
# RESUME
batch_size=8
dropout_rates=0.1
lexicon_window_sizes=9
max_epoch=10

# MSRA
#batch_size=64
#dropout_rates=0.2
#lexicon_window_sizes=4
#max_epoch=5

# weibo
#batch_size=64
#max_epoch=20
#dropout_rates=0.4
#lexicon_window_sizes=5

# ontonotes4
#batch_size=32
#max_epoch=4
#dropout_rates=0.3
#lexicon_window_sizes=4

experts_layers=2
experts_num=1
random_seed=12345
ple_dropout=0.0     # PLE里面使用dropout rate
span_loss_weight=-1 # -1表示不调整span_loss，还是1/3
pactivation=elu  # PLE里面使用的激活函数
use_ff=0  # 是否使用feedforward网络
decay_rate=1.0  # 针对resume和msra使用的学习率衰减

python_command="
python train_bert_mtl_span_attr_boundary.py \
    --pretrain_path /home/qiumengchuan/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
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
    --metric micro_f1
"

if [ $1 == weibo -o $1 == resume ]
then
    maxlen=200
    maxep=10
else
    maxlen=256
    maxep=5
fi

for dpr in ${dropout_rates[*]}
do  
echo "Run dataset $1: dpr=$dpr"
CUDA_VISIBLE_DEVICES=$2 \
$python_command \
--max_length $maxlen \
--max_epoch $maxep \
--dropout_rate $dpr \
--batch_size $batch_size \
--experts_layers $experts_layers \
--experts_num $experts_num \
--random_seed $random_seed \
--ple_dropout $ple_dropout \
--pactivation $pactivation \
--use_ff $use_ff \
--lr_decay $decay_rate
done
