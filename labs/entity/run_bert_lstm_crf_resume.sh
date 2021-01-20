CUDA_VISIBLE_DEVICES=2 \
python train_bert_crf.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --bert_name bert \
    --dataset resume \
    --tagscheme bmoes \
    --compress_seq \
    --use_lstm \
    --use_crf \
    --batch_size 16 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --dropout_rate 0.1 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 200 \
    --max_epoch 10 \
    --optimizer adam \
    --loss ce \
    --adv fgm \
    --dice_alpha 0.6 \
    --metric micro_f1 \

CUDA_VISIBLE_DEVICES=2 \
python train_bert_crf.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --bert_name bert \
    --dataset resume \
    --tagscheme bmoes \
    --compress_seq \
    --use_lstm \
    --use_crf \
    --batch_size 16 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --dropout_rate 0.2 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 200 \
    --max_epoch 10 \
    --optimizer adam \
    --loss ce \
    --adv fgm \
    --dice_alpha 0.6 \
    --metric micro_f1 \

CUDA_VISIBLE_DEVICES=2 \
python train_bert_crf.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --bert_name bert \
    --dataset resume \
    --tagscheme bmoes \
    --compress_seq \
    --use_lstm \
    --use_crf \
    --batch_size 16 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --dropout_rate 0.3 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 200 \
    --max_epoch 10 \
    --optimizer adam \
    --loss ce \
    --adv fgm \
    --dice_alpha 0.6 \
    --metric micro_f1 \

CUDA_VISIBLE_DEVICES=2 \
python train_bert_crf.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --bert_name bert \
    --dataset resume \
    --tagscheme bmoes \
    --compress_seq \
    --use_lstm \
    --use_crf \
    --batch_size 16 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --dropout_rate 0.4 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 200 \
    --max_epoch 10 \
    --optimizer adam \
    --loss ce \
    --adv fgm \
    --dice_alpha 0.6 \
    --metric micro_f1 \

CUDA_VISIBLE_DEVICES=2 \
python train_bert_crf.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --bert_name bert \
    --dataset resume \
    --tagscheme bmoes \
    --compress_seq \
    --use_lstm \
    --use_crf \
    --batch_size 16 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --dropout_rate 0.5 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 200 \
    --max_epoch 10 \
    --optimizer adam \
    --loss ce \
    --adv fgm \
    --dice_alpha 0.6 \
    --metric micro_f1 \

