CUDA_VISIBLE_DEVICES=2 \
python train_supervised_bert_with_dsp.py \
    --pretrain_path ~/NLP/corpus/transformers/google-bert-large-uncased-wwm \
    --model_type entity \
    --dataset semeval \
    --dropout_rate 0.1 \
    --neg_classes [1] \
    --compress_seq \
    --dsp_preprocessed \
    --use_attention \
    --adv none \
    --loss ce \
    --dice_alpha 0.6 \
    --batch_size 16 \
    --lr 2e-5 \
    --bert_lr 2e-5 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 128 \
    --max_dsp_path_length 15 \
    --max_epoch 5 \
    --metric micro_f1 \
    --dsp_tool stanza \
    --optimizer adam 

