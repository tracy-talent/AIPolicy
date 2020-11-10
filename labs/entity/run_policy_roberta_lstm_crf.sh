CUDA_VISIBLE_DEVICES=1 \
python train_bert_crf.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/clue-chinese-roberta-large \
    --bert_name largeroberta \
    --metric micro_f1 \
    --dataset policy \
    --compress_seq \
    --tagscheme bmoes \
    --use_lstm \
    --use_crf \
    --batch_size 8 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --weight_decay 0 \
    --warmup_step 0 \
    --max_length 256 \
    --max_epoch 20 \
    --optimizer adam
