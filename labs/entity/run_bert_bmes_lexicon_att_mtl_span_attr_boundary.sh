#!/bin/bash
# $1: dataset, $2: word2vec_file, $3: GPU id
# RESUME
#batch_size=8
#max_epoch=10
#dropout_rates=0.1
#lexicon_window_sizes=9

# MSRA
batch_size=64
max_epoch=5
dropout_rates=0.2
lexicon_window_sizes=4

#WEIBO
#batch_size=64
#max_epoch=20
#dropout_rates=0.4
#lexicon_window_sizes=5

# ONTONOTES4
#batch_size=32
#max_epoch=3
#dropout_rates=0.3
#lexicon_window_sizes=4

experts_layers=2
experts_num=1
random_seed=12345
ple_dropout=0.1     # PLE里面使用dropout rate
span_loss_weight=-1 # -1表示不调整span_loss，还是1/3
pactivation=gelu  # PLE里面使用的激活函数
use_ff=0  # 是否使用feedforward网络
decay_rate=0.55  # 针对ontonotes4和msra使用的学习率衰减


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
        echo "Run dataset $1: dpr=$dpr, wz=$lws, el=$experts_layers"
        CUDA_VISIBLE_DEVICES=$3 \
        $python_command \
        --word2vec_file /root/qmc/NLP/corpus/embedding/chinese/lexicon/$lexicon2vec \
        --max_length $maxlen \
        --max_epoch $max_epoch \
        --dropout_rate $dpr \
        --lexicon_window_size $lws \
        --batch_size $batch_size \
        --experts_layers $experts_layers \
        --experts_num $experts_num \
        --random_seed $random_seed \
        --ple_dropout $ple_dropout \
        --pactivation $pactivation \
        --use_ff $use_ff \
        --lr_decay $decay_rate
#        --only_test \
#        --ckpt /root/qmc/github/AIPolicy/output/entity/ckpt_3.28_wopinyin/$1_bmoes/bmes3_lexicon_ctb_window13_mtl_span_attr_boundary_ple_bert_relu_crf1e-03_bert3e-05_spanlstm_attrlstm_spancrf_ce_fgm_dpr0.1_micro_f1_1.pth.tar
        done
    done
done