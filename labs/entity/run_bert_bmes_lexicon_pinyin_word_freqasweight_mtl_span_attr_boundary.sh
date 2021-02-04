GPU=0
dropout_rates=(0.1 0.2 0.3 0.4 0.5)
default_dropout=0.2 # 调整lexicon_window_size时使用
lexicon_window_sizes=(4 5 6 7)
default_lexicon_window=5 #调整dropout_rates时使用
python_command="
python train_bert_bmes_lexicon_pinyin_freqasweight_mtl_span_attr_boundary.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --word2vec_file /home/liujian/NLP/corpus/embedding/chinese/lexicon/ctbword_gigachar_mix.710k.50d.bin \
    --pinyin2vec_file /home/liujian/NLP/corpus/pinyin/word2vec/word2vec_num5.1412.50d.vec \
    --word2pinyin_file /home/liujian/NLP/corpus/pinyin/word2pinyin_num5.txt \
    --pinyin_embedding_type word \
    --group_num 3 \
    --model_type ple \
    --dataset $1 \
    --compress_seq \
    --tagscheme bmoes \
    --bert_name bert \
    --span_use_lstm \
    --span_use_crf \
    --attr_use_lstm \
    --batch_size 16 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 200 \
    --max_pinyin_char_length 7 \
    --pinyin_char_embedding_size 50 \
    --max_epoch 10 \
    --optimizer adam \
    --loss ce \
    --adv fgm \
    --metric micro_f1
"

if [ "$2" = "-d" ]; then  # adjust dropout_rate
for param in ${dropout_rates[*]}
do
echo "Run dataset{$1}: dpr=$param,wz=$default_lexicon_window"
CUDA_VISIBLE_DEVICES=${GPU} \
${python_command} \
--dropout_rate ${param} \
--lexicon_window_size ${default_lexicon_window}
done

elif [ "$2" = "-w" ]; then  # adjust lexicon_window_size
for param in ${lexicon_window_sizes[*]}
do
echo "Run dataset{$1}: dpr=$default_dropout,wz=$param"
CUDA_VISIBLE_DEVICES=${GPU} \
${python_command} \
--dropout_rate ${default_dropout} \
--lexicon_window_size ${param}
done

else
	echo "no such parameter: $2"
fi



