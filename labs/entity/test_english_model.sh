CUDA_VISIBLE_DEVICES=0 \
python train_bert_mtl_span_attr_boundary.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/google-bert-base-uncased \
    --model_type ple \
    --dataset conll2003 \
    --only_test \
    --compress_seq \
    --tagscheme bmoes \
    --bert_name bert \
    --span_use_lstm \
    --span_use_crf \
    --attr_use_lstm \
    --soft_label True \
    --batch_size 16 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --dropout_rate 0.3 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 150 \
    --max_epoch 10 \
    --optimizer adam \
    --loss dice \
    --adv fgm \
    --dice_alpha 0.1 \
    --metric micro_f1

