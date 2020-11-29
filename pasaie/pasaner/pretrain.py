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
    if model_name == 'policy_bmoes_bert_crf':
        tag2id = load_vocab(os.path.join(config['path']['ner_dataset'], 'policy/tag2id.bmoes'))
        entity_encoder = encoder.BERTEncoder(
            max_length=256,
            pretrain_path=pretrain_path,
            blank_padding=True,
            bert_name='bert'
        )
        entity_model = model.BILSTM_CRF(
            sequence_encoder=entity_encoder,
            tag2id=tag2id,
            use_lstm=False,
            use_crf=True,
            tagscheme='bmoes'
        )
        entity_model.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
        return entity_model
    elif model_name == 'policy_bmoes_bert_lstm_crf':
        tag2id = load_vocab(os.path.join(config['path']['ner_dataset'], 'policy/tag2id.bmoes'))
        entity_encoder = encoder.BERTEncoder(
            max_length=256,
            pretrain_path=pretrain_path,
            blank_padding=False,
            bert_name='bert'
        )
        entity_model = model.BILSTM_CRF(
            sequence_encoder=entity_encoder,
            tag2id=tag2id,
            use_lstm=True,
            use_crf=True,
            tagscheme='bmoes',
            # compress_seq=False
        )
        entity_model.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
        return entity_model
    else:
        raise NotImplementedError