from . import encoder
from . import model
from . import framework
import torch
import os
import sys
import json
import numpy as np
import logging
from ..tokenization.utils import load_vocab
import configparser

root_url = "https://thunlp.oss-cn-qingdao.aliyuncs.com/"
project_path = '/'.join(os.path.abspath(__file__).replace('\\', '/').split('/')[:-3])
config = configparser.ConfigParser()
config.read(os.path.join(project_path, 'config.ini'))
default_root_path = config['path']['re_ckpt']
# default_root_path = os.path.join(os.getenv('HOME'), '.opennre')

def check_root(root_path=default_root_path):
    if not os.path.exists(root_path):
        os.mkdir(root_path)
        os.mkdir(os.path.join(root_path, 'benchmark'))
        os.mkdir(os.path.join(root_path, 'pretrain'))
        os.mkdir(os.path.join(root_path, 'pretrain/nre'))

def download_wiki80(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'benchmark/relation/wiki80')):
        os.mkdir(os.path.join(root_path, 'benchmark/relation/wiki80'))
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/relation/wiki80') + ' ' + root_url + 'opennre/benchmark/re/wiki80/wiki80_rel2id.json')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/relation/wiki80') + ' ' + root_url + 'opennre/benchmark/re/wiki80/wiki80_train.txt')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/relation/wiki80') + ' ' + root_url + 'opennre/benchmark/re/wiki80/wiki80_val.txt')

def download_tacred(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'benchmark/relation/tacred')):
        os.mkdir(os.path.join(root_path, 'benchmark/relation/tacred'))
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/relation/tacred') + ' ' + root_url + 'opennre/benchmark/re/tacred/tacred_rel2id.json')
        logging.info('Due to copyright limits, we only provide rel2id for TACRED. Please download TACRED manually and convert the data to OpenNRE format if needed.')

def download_nyt10(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'benchmark/relation/nyt10')):
        os.mkdir(os.path.join(root_path, 'benchmark/relation/nyt10'))
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/relation/nyt10') + ' ' + root_url + 'opennre/benchmark/re/nyt10/nyt10_rel2id.json')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/relation/nyt10') + ' ' + root_url + 'opennre/benchmark/re/nyt10/nyt10_train.txt')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/relation/nyt10') + ' ' + root_url + 'opennre/benchmark/re/nyt10/nyt10_test.txt')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/relation/nyt10') + ' ' + root_url + 'opennre/benchmark/re/nyt10/nyt10_val.txt')

def download_wiki_distant(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'benchmark/relation/wiki_distant')):
        os.mkdir(os.path.join(root_path, 'benchmark/relation/wiki_distant'))
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/relation/wiki_distant') + ' ' + root_url + 'opennre/benchmark/re/wiki_distant/wiki_distant_rel2id.json')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/relation/wiki_distant') + ' ' + root_url + 'opennre/benchmark/re/wiki_distant/wiki_distant_train.txt')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/relation/wiki_distant') + ' ' + root_url + 'opennre/benchmark/re/wiki_distant/wiki_distant_test.txt')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/relation/wiki_distant') + ' ' + root_url + 'opennre/benchmark/re/wiki_distant/wiki_distant_val.txt')

def download_semeval(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'benchmark/relation/semeval')):
        os.mkdir(os.path.join(root_path, 'benchmark/relation/semeval'))
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/relation/semeval') + ' ' + root_url + 'opennre/benchmark/re/semeval/semeval_rel2id.json')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/relation/semeval') + ' ' + root_url + 'opennre/benchmark/re/semeval/semeval_train.txt')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/relation/semeval') + ' ' + root_url + 'opennre/benchmark/re/semeval/semeval_test.txt')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/relation/semeval') + ' ' + root_url + 'opennre/benchmark/re/semeval/semeval_val.txt')

def download_glove(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'pretrain/glove')):
        os.mkdir(os.path.join(root_path, 'pretrain/glove'))
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/glove') +  ' ' + root_url + 'opennre/pretrain/glove/glove.6B.50d_mat.npy')
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/glove') +  ' ' + root_url + 'opennre/pretrain/glove/glove.6B.50d_word2id.json')

def download_bert_base_uncased(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'pretrain/bert-base-uncased')):
        os.mkdir(os.path.join(root_path, 'pretrain/bert-base-uncased'))
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/bert-base-uncased') + ' ' + root_url + 'opennre/pretrain/bert-base-uncased/config.json')
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/bert-base-uncased') + ' ' + root_url + 'opennre/pretrain/bert-base-uncased/pytorch_model.bin')
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/bert-base-uncased') + ' ' + root_url + 'opennre/pretrain/bert-base-uncased/vocab.txt')

def download_pretrain(model_name, root_path=default_root_path):
    ckpt = os.path.join(root_path, 'pretrain/nre/' + model_name + '.pth.tar')
    if not os.path.exists(ckpt):
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/nre')  + ' ' + root_url + 'opennre/pretrain/nre/' + model_name + '.pth.tar')

def download(name, root_path=default_root_path):
    check_root()
    if name == 'nyt10':
        download_nyt10(root_path=root_path)
    elif name == 'wiki_distant':
        download_wiki_distant(root_path=root_path)
    elif name == 'semeval':
        download_semeval(root_path=root_path)
    elif name == 'wiki80':
        download_wiki80(root_path=root_path)
    elif name == 'tacred':
        download_tacred(root_path=root_path)
    elif name == 'glove':
        download_glove(root_path=root_path)
    elif name == 'bert_base_uncased':
        download_bert_base_uncased(root_path=root_path)
    else:
        raise Exception('Cannot find corresponding data.')

def load_json(file_name):
    with open(file_name) as f:
        return json.load(f)
        
def get_model(model_name, pretrain_path=config['plm']['hfl-chinese-bert-wwm-ext'], root_path=default_root_path):
    check_root()
    ckpt = os.path.join(config['path']['re_ckpt'], model_name + '.pth.tar')
    print(ckpt)
    dataset_name = model_name.split('/')[0]
    if 'policy' in dataset_name:
        rel2id = load_json(os.path.join(config['path']['re_dataset'], f'{dataset_name}/{dataset_name}_rel2id.json'))
        tag2id = load_json(os.path.join(config['path']['re_dataset'], f'{dataset_name}/{dataset_name}_tag2id.json'))
        if 'dsp' in model_name:
            sentence_encoder = encoder.BERTEntityWithDSPEncoder(
                pretrain_path=pretrain_path, max_length=256, max_dsp_path_length=15, compress_seq=False,
                use_attention=('attention' in model_name), mask_entity=False, blank_padding=False, tag2id=None)
        elif 'bert_entity' in model_name:
            sentence_encoder = encoder.BERTEntityEncoder(
                max_length=256, pretrain_path=pretrain_path, mask_entity=False, 
                blank_padding=False, tag2id=tag2id)
        else:
            sentence_encoder = encoder.BERTEncoder(
                pretrain_path=pretrain_path, max_length=256, blank_padding=False)
        relation_model = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        relation_model.load_state_dict(torch.load(ckpt, map_location='cpu')['model'])
        relation_model.eval()
        return relation_model
    elif model_name == 'wiki80_cnn_softmax':
        download_pretrain(model_name, root_path=root_path)
        download('glove', root_path=root_path)
        download('wiki80', root_path=root_path)
        wordi2d = json.load(open(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_word2id.json')))
        word2vec = np.load(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_mat.npy'))
        rel2id = json.load(open(os.path.join(root_path, 'benchmark/relation/wiki80/wiki80_rel2id.json')))
        sentence_encoder = encoder.CNNEncoder(token2id=wordi2d,
                                                     max_length=40,
                                                     word_size=50,
                                                     position_size=5,
                                                     hidden_size=230,
                                                     blank_padding=True,
                                                     kernel_size=3,
                                                     padding_size=1,
                                                     word2vec=word2vec,
                                                     dropout=0.5)
        m = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        m.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
        return m
    elif model_name in ['wiki80_bert_softmax', 'wiki80_bertentity_softmax']:
        download_pretrain(model_name, root_path=root_path)
        download('bert_base_uncased', root_path=root_path)
        download('wiki80', root_path=root_path)
        rel2id = json.load(open(os.path.join(root_path, 'benchmark/relation/wiki80/wiki80_rel2id.json')))
        if 'entity' in model_name:
            sentence_encoder = encoder.BERTEntityEncoder(
                max_length=80, pretrain_path=os.path.join(root_path, 'pretrain/bert-base-uncased'))
        else:
            sentence_encoder = encoder.BERTEncoder(
                max_length=80, pretrain_path=os.path.join(root_path, 'pretrain/bert-base-uncased'))
        m = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        m.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
        return m
    elif model_name in ['tacred_bert_softmax', 'tacred_bertentity_softmax']:
        download_pretrain(model_name, root_path=root_path)
        download('bert_base_uncased', root_path=root_path)
        download('tacred', root_path=root_path)
        rel2id = json.load(open(os.path.join(root_path, 'benchmark/relation/tacred/tacred_rel2id.json')))
        if 'entity' in model_name:
            sentence_encoder = encoder.BERTEntityEncoder(
                max_length=80, pretrain_path=os.path.join(root_path, 'pretrain/bert-base-uncased'))
        else:
            sentence_encoder = encoder.BERTEncoder(
                max_length=80, pretrain_path=os.path.join(root_path, 'pretrain/bert-base-uncased'))
        m = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        m.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
        return m
    else:
        raise NotImplementedError
