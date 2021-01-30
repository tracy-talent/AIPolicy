CUDA_VISIBLE_DEVICES=2 \
python train_bert_bmes_lexicon_pinyin_mtl_span_attr_boundary.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --word2vec_file /home/liujian/NLP/corpus/embedding/chinese/lexicon/ctbword_gigachar_mix.710k.50d.bin \
    --word2pinyin_file /home/liujian/NLP/corpus/pinyin/word2pinyin_num5.txt \
    --pinyin_embedding_type att_together_add \
    --model_type ple \
    --dataset resume \
    --compress_seq \
    --tagscheme bmoes \
    --bert_name bert \
    --span_use_lstm \
    --span_use_crf \
    --attr_use_lstm \
    --soft_label True \
    --batch_size 10 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --dropout_rate 0.1 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 200 \
    --max_pinyin_num_of_token 10 \
    --max_pinyin_char_length 7 \
    --lexicon_window_size 5 \
    --pinyin_word_embedding_size 50 \
    --pinyin_char_embedding_size 50 \
    --max_epoch 10 \
    --optimizer adam \
    --loss ce \
    --adv fgm \
    --metric micro_f1

CUDA_VISIBLE_DEVICES=2 \
python train_bert_bmes_lexicon_pinyin_mtl_span_attr_boundary.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --word2vec_file /home/liujian/NLP/corpus/embedding/chinese/lexicon/ctbword_gigachar_mix.710k.50d.bin \
    --word2pinyin_file /home/liujian/NLP/corpus/pinyin/word2pinyin_num5.txt \
    --pinyin_embedding_type att_together_add \
    --model_type ple \
    --dataset resume \
    --compress_seq \
    --tagscheme bmoes \
    --bert_name bert \
    --span_use_lstm \
    --span_use_crf \
    --attr_use_lstm \
    --soft_label True \
    --batch_size 10 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --dropout_rate 0.2 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 200 \
    --max_pinyin_num_of_token 10 \
    --max_pinyin_char_length 7 \
    --lexicon_window_size 5 \
    --pinyin_word_embedding_size 50 \
    --pinyin_char_embedding_size 50 \
    --max_epoch 10 \
    --optimizer adam \
    --loss ce \
    --adv fgm \
    --metric micro_f1

CUDA_VISIBLE_DEVICES=2 \
python train_bert_bmes_lexicon_pinyin_mtl_span_attr_boundary.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --word2vec_file /home/liujian/NLP/corpus/embedding/chinese/lexicon/ctbword_gigachar_mix.710k.50d.bin \
    --word2pinyin_file /home/liujian/NLP/corpus/pinyin/word2pinyin_num5.txt \
    --pinyin_embedding_type att_together_add \
    --model_type ple \
    --dataset resume \
    --compress_seq \
    --tagscheme bmoes \
    --bert_name bert \
    --span_use_lstm \
    --span_use_crf \
    --attr_use_lstm \
    --soft_label True \
    --batch_size 10 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --dropout_rate 0.3 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 200 \
    --max_pinyin_num_of_token 10 \
    --max_pinyin_char_length 7 \
    --lexicon_window_size 5 \
    --pinyin_word_embedding_size 50 \
    --pinyin_char_embedding_size 50 \
    --max_epoch 10 \
    --optimizer adam \
    --loss ce \
    --adv fgm \
    --metric micro_f1

CUDA_VISIBLE_DEVICES=2 \
python train_bert_bmes_lexicon_pinyin_mtl_span_attr_boundary.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --word2vec_file /home/liujian/NLP/corpus/embedding/chinese/lexicon/ctbword_gigachar_mix.710k.50d.bin \
    --word2pinyin_file /home/liujian/NLP/corpus/pinyin/word2pinyin_num5.txt \
    --pinyin_embedding_type att_together_add \
    --model_type ple \
    --dataset resume \
    --compress_seq \
    --tagscheme bmoes \
    --bert_name bert \
    --span_use_lstm \
    --span_use_crf \
    --attr_use_lstm \
    --soft_label True \
    --batch_size 10 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --dropout_rate 0.4 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 200 \
    --max_pinyin_num_of_token 10 \
    --max_pinyin_char_length 7 \
    --lexicon_window_size 5 \
    --pinyin_word_embedding_size 50 \
    --pinyin_char_embedding_size 50 \
    --max_epoch 10 \
    --optimizer adam \
    --loss ce \
    --adv fgm \
    --metric micro_f1

CUDA_VISIBLE_DEVICES=2 \
python train_bert_bmes_lexicon_pinyin_mtl_span_attr_boundary.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --word2vec_file /home/liujian/NLP/corpus/embedding/chinese/lexicon/ctbword_gigachar_mix.710k.50d.bin \
    --word2pinyin_file /home/liujian/NLP/corpus/pinyin/word2pinyin_num5.txt \
    --pinyin_embedding_type att_together_add \
    --model_type ple \
    --dataset resume \
    --compress_seq \
    --tagscheme bmoes \
    --bert_name bert \
    --span_use_lstm \
    --span_use_crf \
    --attr_use_lstm \
    --soft_label True \
    --batch_size 10 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --dropout_rate 0.5 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 200 \
    --max_pinyin_num_of_token 10 \
    --max_pinyin_char_length 7 \
    --lexicon_window_size 5 \
    --pinyin_word_embedding_size 50 \
    --pinyin_char_embedding_size 50 \
    --max_epoch 10 \
    --optimizer adam \
    --loss ce \
    --adv fgm \
    --metric micro_f1

