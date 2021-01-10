CUDA_VISIBLE_DEVICES=0 \
python train_bert_wlf_mtl_span_attr_boundary.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --word2vec_file /home/liujian/NLP/corpus/embedding/chinese/lexicon/sgns_merge_word.1293k.300d.vec \
    --model_type attention \
    --dataset weibo \
    --compress_seq \
    --tagscheme bmoes \
    --bert_name bert \
    --span_use_lstm \
    --span_use_crf \
    --attr_use_lstm \
    --soft_label True \
    --batch_size 4 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --dropout_rate 0.3 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 200 \
    --max_epoch 10 \
    --optimizer adam \
    --loss ce \
    --adv none \
    --dice_alpha 0.8 \
    --metric micro_f1
