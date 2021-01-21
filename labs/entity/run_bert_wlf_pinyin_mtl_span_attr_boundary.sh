CUDA_VISIBLE_DEVICES=2 \
python train_bert_wlf_pinyin_mtl_span_attr_boundary.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --word2vec_file /home/liujian/NLP/corpus/embedding/chinese/lexicon/ctbword_gigachar_mix.710k.50d.vec \
    --custom_dict /home/liujian/github/AIPolicy/input/benchmark/entity/weibo/custom_dict.txt \
    --word2pinyin_file /home/liujian/NLP/corpus/pinyin/word2pinyin_num.txt \
    --pinyin_embedding_type char_multiconv \
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
    --max_pinyin_num_of_token 10 \
    --max_pinyin_char_length 7 \
    --pinyin_word_embedding_size 50 \
    --pinyin_char_embedding_size 50 \
    --max_epoch 5 \
    --optimizer adam \
    --loss ce \
    --adv fgm \
    --dice_alpha 0.1 \
    --metric micro_f1

