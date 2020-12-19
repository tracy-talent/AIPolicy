CUDA_VISIBLE_DEVICES=0 \
python train_bert_crf.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --dataset policy \
    --tagscheme bmoes \
    --compress_seq \
    --use_crf \
    --batch_size 16 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --weight_decay 0 \
    --early_stopping_step 3 \
    --warmup_step 0 \
    --max_length 256 \
    --max_epoch 20 \
    --optimizer adam \
    --loss dice \
    --adv fgm \
    --dice_alpha 0.6 \
    --metric micro_f1 \
