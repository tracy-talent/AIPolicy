CUDA_VISIBLE_DEVICES=3 \
python train_bert_mrc_span_mtl.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --bert_name bert \
    --dataset policy \
    --compress_seq \
    --use_lstm \
    --tagscheme bmoes \
    --batch_size 16 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 320 \
    --max_epoch 20 \
    --dropout_rate 0.1 \
    --loss ce \
    --dice_alpha 0.6 \
    --adv none \
    --optimizer adam \
    --metric micro_f1 

