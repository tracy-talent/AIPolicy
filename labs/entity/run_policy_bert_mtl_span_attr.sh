CUDA_VISIBLE_DEVICES=0 \
python train_bert_mtl_span_attr.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --metric micro_f1 \
    --dataset policy \
    --compress_seq \
    --tagscheme bmoes \
    --bert_name bert \
    --span_use_lstm \
    --span_use_crf \
    --attr_use_lstm \
    --batch_size 16 \
    --lr 1e-3 \
    --bert_lr 1e-4 \
    --weight_decay 0 \
    --warmup_step 30 \
    --max_length 256 \
    --max_epoch 200 \
    --optimizer adam
