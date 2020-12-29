CUDA_VISIBLE_DEVICES=2 \
python train_supervised_bert.py \
    --pretrain_path ~/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --pooler entity \
    --metric micro_f1 \
    --dataset test-policy \
    --dropout_rate 0.5 \
    --neg_classes [1] \
    --compress_seq \
    --adv fgm \
    --loss dice \
    --dice_alpha 0.6 \
    --batch_size 12 \
    --lr 1e-5 \
    --bert_lr 1e-5 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 256 \
    --max_epoch 20 \
    --optimizer adam \
    --use_sampler 

