CUDA_VISIBLE_DEVICES=3 \
python train_bert_span_ner.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --bert_name bert \
    --model multi \
    --dataset resume \
    --compress_seq \
    --use_lstm \
    --tagscheme bmoes \
    --batch_size 12 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 200 \
    --max_epoch 10 \
    --dropout_rate 0.1 \
    --loss ce \
    --dice_alpha 0.6 \
    --adv fgm \
    --optimizer adam \
    --metric micro_f1

CUDA_VISIBLE_DEVICES=3 \
python train_bert_span_ner.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --bert_name bert \
    --model multi \
    --dataset resume \
    --compress_seq \
    --use_lstm \
    --tagscheme bmoes \
    --batch_size 12 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 200 \
    --max_epoch 10 \
    --dropout_rate 0.2 \
    --loss ce \
    --dice_alpha 0.6 \
    --adv fgm \
    --optimizer adam \
    --metric micro_f1

CUDA_VISIBLE_DEVICES=3 \
python train_bert_span_ner.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --bert_name bert \
    --model multi \
    --dataset resume \
    --compress_seq \
    --use_lstm \
    --tagscheme bmoes \
    --batch_size 12 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 200 \
    --max_epoch 10 \
    --dropout_rate 0.3 \
    --loss ce \
    --dice_alpha 0.6 \
    --adv fgm \
    --optimizer adam \
    --metric micro_f1

CUDA_VISIBLE_DEVICES=3 \
python train_bert_span_ner.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --bert_name bert \
    --model multi \
    --dataset resume \
    --compress_seq \
    --use_lstm \
    --tagscheme bmoes \
    --batch_size 12 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 200 \
    --max_epoch 10 \
    --dropout_rate 0.4 \
    --loss ce \
    --dice_alpha 0.6 \
    --adv fgm \
    --optimizer adam \
    --metric micro_f1

CUDA_VISIBLE_DEVICES=3 \
python train_bert_span_ner.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --bert_name bert \
    --model multi \
    --dataset resume \
    --compress_seq \
    --use_lstm \
    --tagscheme bmoes \
    --batch_size 12 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 200 \
    --max_epoch 10 \
    --dropout_rate 0.5 \
    --loss ce \
    --dice_alpha 0.6 \
    --adv fgm \
    --optimizer adam \
    --metric micro_f1

