CUDA_VISIBLE_DEVICES=0 \
python train_wlf_bilstm_crf.py \
    --pretrain_path /home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --metric micro_f1 \
    --dataset policy \
    --compress_seq \
    --char2vec_file /home/liujian/NLP/corpus/embedding/chinese/lexicon/ctbword_gigachar_mix.710k.50d.vec \
    --word2vec_file /home/liujian/NLP/corpus/embedding/chinese/lexicon/ctbword_gigachar_mix.710k.50d.vec \
    --tagscheme bmoes \
    --use_lstm \
    --use_crf \
    --batch_size 16 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --weight_decay 0 \
    --warmup_step 0 \
    --max_length 256 \
    --max_epoch 30 \
    --optimizer adam
