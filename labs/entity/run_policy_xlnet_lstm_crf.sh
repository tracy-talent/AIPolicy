CUDA_VISIBLE_DEVICES=0 \
python train_xlnet_crf.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/clue-chinese-xlnet-large \
    --bert_name largexlnet \
    --metric micro_f1 \
    --dataset policy \
    --tagscheme bmoes \
    --use_lstm \
    --use_crf \
    --batch_size 4 \
    --lr 1e-3 \
    --bert_lr 1e-5 \
    --weight_decay 0 \
    --warmup_step 0 \
    --max_length 256 \
    --max_epoch 20 \
    --optimizer adam
