CUDA_VISIBLE_DEVICES=3 \
python train_supervised_bert.py \
    --pretrain_path ~/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --pooler entity \
    --metric micro_f1 \
    --dataset test-policy \
    --batch_size 32 \
    --lr 3e-5 \
    --weight_decay 0 \
    --warmup_step 0 \
    --max_length 256 \
    --max_epoch 20 \
    --optimizer adamw \
    --use_sampler 

