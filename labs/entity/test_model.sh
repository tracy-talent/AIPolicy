CUDA_VISIBLE_DEVICES=0 \
python train_bert_wlf_mtl_span_attr_boundary.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --word2vec_file /home/liujian/NLP/corpus/embedding/chinese/lexicon/ctbword_gigachar_mix.710k.50d.vec \
    --custom_dict /home/liujian/github/AIPolicy/input/benchmark/entity/weibo/custom_dict.txt \
    --model_type ple \
    --dataset weibo \
    --compress_seq \
    --tagscheme bmoes \
    --bert_name bert \
    --span_use_lstm \
    --span_use_crf \
    --attr_use_lstm \
    --soft_label True \
    --batch_size 8 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --dropout_rate 0.3 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 200 \
    --max_epoch 5 \
    --optimizer adam \
    --loss dice \
    --adv fgm \
    --dice_alpha 0.7 \
    --metric micro_f1

