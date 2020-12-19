CUDA_VISIBLE_DEVICES=0 \
python train_xlnet_crf.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/clue-chinese-xlnet-large \
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
    --early_stopping_step 3 \
    --warmup_step 0 \
    --max_length 256 \
    --max_epoch 20 \
    --optimizer adam \
    --metric micro_f1 \
