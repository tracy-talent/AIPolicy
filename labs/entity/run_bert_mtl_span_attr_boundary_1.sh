CUDA_VISIBLE_DEVICES=3 \
python train_bert_mtl_span_attr_boundary.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --model_type attention \
    --dataset ontonotes4 \
    --compress_seq \
    --tagscheme bmoes \
    --bert_name bert \
    --span_use_lstm \
    --span_use_crf \
    --attr_use_lstm \
    --soft_label True \
    --batch_size 14 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --dropout_rate 0.3 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 256 \
    --max_epoch 5 \
    --optimizer adam \
    --loss dice \
    --adv fgm \
    --dice_alpha 0.04 \
    --metric micro_f1

CUDA_VISIBLE_DEVICES=3 \
python train_bert_mtl_span_attr_boundary.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --model_type attention \
    --dataset ontonotes4 \
    --compress_seq \
    --tagscheme bmoes \
    --bert_name bert \
    --span_use_lstm \
    --span_use_crf \
    --attr_use_lstm \
    --soft_label True \
    --batch_size 14 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --dropout_rate 0.3 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 256 \
    --max_epoch 5 \
    --optimizer adam \
    --loss dice \
    --adv fgm \
    --dice_alpha 0.03 \
    --metric micro_f1

CUDA_VISIBLE_DEVICES=3 \
python train_bert_mtl_span_attr_boundary.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --model_type attention \
    --dataset ontonotes4 \
    --compress_seq \
    --tagscheme bmoes \
    --bert_name bert \
    --span_use_lstm \
    --span_use_crf \
    --attr_use_lstm \
    --soft_label True \
    --batch_size 14 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --dropout_rate 0.3 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 256 \
    --max_epoch 5 \
    --optimizer adam \
    --loss dice \
    --adv fgm \
    --dice_alpha 0.02 \
    --metric micro_f1

CUDA_VISIBLE_DEVICES=3 \
python train_bert_mtl_span_attr_boundary.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --model_type attention \
    --dataset ontonotes4 \
    --compress_seq \
    --tagscheme bmoes \
    --bert_name bert \
    --span_use_lstm \
    --span_use_crf \
    --attr_use_lstm \
    --soft_label True \
    --batch_size 14 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --dropout_rate 0.3 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 256 \
    --max_epoch 5 \
    --optimizer adam \
    --loss dice \
    --adv fgm \
    --dice_alpha 0.01 \
    --metric micro_f1
