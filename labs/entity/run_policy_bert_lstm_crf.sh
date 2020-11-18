CUDA_VISIBLE_DEVICES=2 \
python train_bert_crf.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --bert_name bert \
    --metric micro_f1 \
    --dataset policy \
    --tagscheme bmoes \
    --compress_seq \
    --use_lstm \
    --use_crf \
    --batch_size 12 \
    --lr 1e-4 \
    --bert_lr 2e-5 \
    --weight_decay 0.01 \
    --warmup_step 0 \
    --max_length 256 \
    --max_epoch 20 \
    --adv fgm \
    --optimizer adam
