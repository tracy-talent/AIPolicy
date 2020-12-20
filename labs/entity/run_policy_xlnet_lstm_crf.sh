CUDA_VISIBLE_DEVICES=2 \
python train_xlnet_crf.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-xlnet-base \
    --bert_name largexlnet \
    --dataset policy \
    --compress_seq \
    --tagscheme bmoes \
    --use_lstm \
    --use_crf \
    --batch_size 8 \
    --lr 1e-3 \
    --bert_lr 1e-5 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 256 \
    --max_epoch 40 \
    --optimizer adam \
    --metric micro_f1 \
