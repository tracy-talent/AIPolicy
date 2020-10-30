CUDA_VISIBLE_DEVICES=2 \
python train_bert_mtl_span_attr.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --metric micro_f1 \
    --dataset policy \
    --tagscheme bmoes \
    --lstm_num 1 \
    --span_use_crf \
    --attr_use_crf \
    --batch_size 16 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --weight_decay 0 \
    --warmup_step 0 \
    --max_length 256 \
    --max_epoch 40 \
    --optimizer adam
