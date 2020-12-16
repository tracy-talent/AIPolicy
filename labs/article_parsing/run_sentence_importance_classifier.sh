CUDA_VISIBLE_DEVICES=2 \
python train_sentence_importance_classifier.py \
    --pretrain_path ~/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --encoder bert \
    --bert_name bert \
    --model textcnn \
    --metric micro_f1 \
    --dataset sentence_importance_judgement \
    --batch_size 16 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --weight_decay 0 \
    --warmup_step 0 \
    --max_length 256 \
    --max_epoch 20 \
    --optimizer adam \
    --adv fgm \
    --dice_alpha 0.6 \
    --loss bce \
    --use_sampler Fasle \
    --compress_seq True


# --pretrain_path ~/qmc_space/NLP/corpus/embedding/chinese/lexicon/gigaword_chn.all.a2b.uni.11k.50d.vec \
#    --pretrain_path ~/qmc_space/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \