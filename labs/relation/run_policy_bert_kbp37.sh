CUDA_VISIBLE_DEVICES=3 \
python train_supervised_bert_with_dsp.py \
    --pretrain_path ~/NLP/corpus/transformers/google-bert-large-uncased-wwm \
    --model_type entity \
    --dataset kbp37 \
    --dropout_rate 0.1 \
    --neg_classes [10] \
    --compress_seq \
    --dsp_preprocessed \
    --use_attention \
    --adv none \
    --loss ce \
    --dice_alpha 0.6 \
    --batch_size 10 \
    --lr 2e-5 \
    --bert_lr 2e-5 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 200 \
    --max_dsp_path_length 22 \
    --max_epoch 10 \
    --metric micro_f1 \
    --dsp_tool stanza \
    --optimizer adam 

CUDA_VISIBLE_DEVICES=3 \
python train_supervised_bert_with_dsp.py \
    --pretrain_path ~/NLP/corpus/transformers/google-bert-large-uncased-wwm \
    --model_type entity \
    --dataset kbp37 \
    --dropout_rate 0.2 \
    --neg_classes [10] \
    --compress_seq \
    --dsp_preprocessed \
    --use_attention \
    --adv none \
    --loss ce \
    --dice_alpha 0.6 \
    --batch_size 10 \
    --lr 2e-5 \
    --bert_lr 2e-5 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 200 \
    --max_dsp_path_length 22 \
    --max_epoch 10 \
    --metric micro_f1 \
    --dsp_tool stanza \
    --optimizer adam 

CUDA_VISIBLE_DEVICES=3 \
python train_supervised_bert_with_dsp.py \
    --pretrain_path ~/NLP/corpus/transformers/google-bert-large-uncased-wwm \
    --model_type entity \
    --dataset kbp37 \
    --dropout_rate 0.3 \
    --neg_classes [10] \
    --compress_seq \
    --dsp_preprocessed \
    --use_attention \
    --adv none \
    --loss ce \
    --dice_alpha 0.6 \
    --batch_size 10 \
    --lr 2e-5 \
    --bert_lr 2e-5 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 200 \
    --max_dsp_path_length 22 \
    --max_epoch 10 \
    --metric micro_f1 \
    --dsp_tool stanza \
    --optimizer adam 

CUDA_VISIBLE_DEVICES=3 \
python train_supervised_bert_with_dsp.py \
    --pretrain_path ~/NLP/corpus/transformers/google-bert-large-uncased-wwm \
    --model_type entity \
    --dataset kbp37 \
    --dropout_rate 0.4 \
    --neg_classes [10] \
    --compress_seq \
    --dsp_preprocessed \
    --use_attention \
    --adv none \
    --loss ce \
    --dice_alpha 0.6 \
    --batch_size 10 \
    --lr 2e-5 \
    --bert_lr 2e-5 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 200 \
    --max_dsp_path_length 22 \
    --max_epoch 10 \
    --metric micro_f1 \
    --dsp_tool stanza \
    --optimizer adam 

CUDA_VISIBLE_DEVICES=3 \
python train_supervised_bert_with_dsp.py \
    --pretrain_path ~/NLP/corpus/transformers/google-bert-large-uncased-wwm \
    --model_type entity \
    --dataset kbp37 \
    --dropout_rate 0.5 \
    --neg_classes [10] \
    --compress_seq \
    --dsp_preprocessed \
    --use_attention \
    --adv none \
    --loss ce \
    --dice_alpha 0.6 \
    --batch_size 10 \
    --lr 2e-5 \
    --bert_lr 2e-5 \
    --weight_decay 0 \
    --early_stopping_step 0 \
    --warmup_step 0 \
    --max_length 200 \
    --max_dsp_path_length 22 \
    --max_epoch 10 \
    --metric micro_f1 \
    --dsp_tool stanza \
    --optimizer adam 
