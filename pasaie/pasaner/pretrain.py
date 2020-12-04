"""
 Author: liujian
 Date: 2020-10-29 14:55:41
 Last Modified by: liujian
 Last Modified time: 2020-10-29 14:55:41
"""

import os

import torch
import configparser

from . import encoder, model
from ..tokenization.utils import load_vocab

project_path = '/'.join(os.path.abspath(__file__).replace('\\', '/').split('/')[:-3])
config = configparser.ConfigParser()
config.read(os.path.join(project_path, 'config.ini'))
default_root_path = config['path']['ner_ckpt']

def get_model(model_name, pretrain_path=config['plm']['hfl-chinese-bert-wwm-ext'], root_path=default_root_path):
    ckpt = os.path.join(config['path']['ner_ckpt'], model_name + '.pth.tar')
    print(ckpt)
    dataset_name = model_name.split('/')[0]
    if dataset_name.endswith('_bmoes'):
        dataset_name = dataset_name[:-6]
    tag2id = load_vocab(os.path.join(config['path']['ner_dataset'], f'{dataset_name}/tag2id.bmoes'))
    if 'multi_bert' in model_name:
        entity_encoder = encoder.BERT_BILSTM_Encoder(
            max_length=256,
            pretrain_path=pretrain_path,
            bert_name='bert',
            use_lstm=False,
            compress_seq=False,
            blank_padding=False  # no padding for inference, otherwise logits size may not equal to mask size during crf decode
        )
        entity_model = model.Span_Pos_CLS(
            sequence_encoder=entity_encoder, 
            tag2id=tag2id, 
            use_lstm=True, 
            compress_seq=False, 
            soft_label=True,
        )
        entity_model.load_state_dict(torch.load(ckpt, map_location='cpu')['model'])
        return entity_model
    elif 'single_bert' in model_name:
        entity_encoder = encoder.BERT_BILSTM_Encoder(
            max_length=256,
            pretrain_path=pretrain_path,
            bert_name='bert',
            use_lstm=True,
            compress_seq=False,
            blank_padding=False  # no padding for inference, otherwise logits size may not equal to mask size during crf decode
        )
        entity_model = model.Span_Cat_CLS(
            sequence_encoder=entity_encoder, 
            tag2id=tag2id, 
            ffn_hidden_size=150,
            width_embedding_size=150,
            max_span=10,
        )
        entity_model.load_state_dict(torch.load(ckpt, map_location='cpu')['model'])
        return entity_model
    elif 'bert' in model_name and 'crf' in model_name:
        entity_encoder = encoder.BERTEncoder(
            max_length=256,
            pretrain_path=pretrain_path,
            bert_name='bert',
            blank_padding=False  # no padding for inference, otherwise logits size may not equal to mask size during crf decode
        )
        entity_model = model.BILSTM_CRF(
            sequence_encoder=entity_encoder,
            tag2id=tag2id,
            use_lstm=('lstm' in model_name),
            use_crf=True,
            tagscheme='bmoes',
            compress_seq=False
        )
        entity_model.load_state_dict(torch.load(ckpt, map_location='cpu')['model'])
        return entity_model
    else:
        raise NotImplementedError