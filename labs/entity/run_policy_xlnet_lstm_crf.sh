CUDA_VISIBLE_DEVICES=0 \
python train_xlnet_crf.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-xlnet-base \
    --bert_name xlnet \
    --metric micro_f1 \
    --dataset policy \
    --compress_seq True \
    --tagscheme bmoes \
    --use_lstm \
    --use_crf \
    --batch_size 16 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --weight_decay 0 \
    --warmup_step 0 \
    --max_length 256 \
    --max_epoch 40 \
    --optimizer adam
