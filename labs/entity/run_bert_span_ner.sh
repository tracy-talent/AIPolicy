CUDA_VISIBLE_DEVICES=0 \
python train_bert_span_ner.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --bert_name bert \
    --model multi \
    --dataset weibo \
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
    --max_epoch 20 \
    --ffn_hidden_size 150 \
    --width_embedding_size 150 \
    --dropout_rate 0.1 \
    --max_span 10 \
    --soft_label False \
    --use_mtl_autoweighted_loss \
    --loss dice \
    --dice_alpha 0.6 \
    --adv fgm \
    --optimizer adam \
    --metric micro_f1

