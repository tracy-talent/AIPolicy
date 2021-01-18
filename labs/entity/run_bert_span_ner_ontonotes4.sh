CUDA_VISIBLE_DEVICES=1 \
python train_bert_span_ner.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --bert_name bert \
    --model multi \
    --dataset ontonotes4 \
    --compress_seq \
    --use_lstm \
    --tagscheme bmoes \
    --batch_size 10 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 256 \
    --max_epoch 5 \
    --dropout_rate 0.5 \
    --loss ce \
    --dice_alpha 0.6 \
    --adv fgm \
    --optimizer adam \
    --metric micro_f1

