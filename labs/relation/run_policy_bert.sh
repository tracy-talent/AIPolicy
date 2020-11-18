CUDA_VISIBLE_DEVICES=0 \
python train_supervised_bert.py \
    --pretrain_path ~/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --pooler entity \
    --metric micro_f1 \
    --dataset test-policy \
    --compress_seq \
    --loss dice \
    --adv fgm \
    --batch_size 12 \
    --lr 1e-3 \
    --bert_lr 2e-5 \
    --weight_decay 0 \
    --warmup_step 0 \
    --max_length 256 \
    --max_epoch 20 \
    --optimizer adam \
    --use_sampler 

