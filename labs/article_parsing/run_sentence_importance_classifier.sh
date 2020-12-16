CUDA_VISIBLE_DEVICES=3 \
python train_sentence_importance_classifier.py \
    --pretrain_path ~/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
    --encoder bert \
    --bert_name bert \
    --model textcnn \
    --metric micro_f1 \
    --dataset sentence_importance_judgement \
    --neg_classes [0] \
    --batch_size 15 \
    --lr 1e-3 \
    --bert_lr 3e-5 \
    --weight_decay 0 \
    --warmup_step 10 \
    --max_length 256 \
    --max_epoch 20 \
    --optimizer adamw \
    --adv pgd \
    --dice_alpha 0.6 \
    --loss pwbce \
    --compress_seq \
    --use_sampler \


#    --pretrain_path ~/qmc_space/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext \
#    --pretrain_path ~/qmc_space/NLP/corpus/embedding/chinese/lexicon/gigaword_chn.all.a2b.uni.11k.50d.vec \
