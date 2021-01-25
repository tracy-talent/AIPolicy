"""
 Author: liujian
 Date: 2021-01-11 23:32:03
 Last Modified by: liujian
 Last Modified time: 2021-01-11 23:32:03
"""

from gensim.models import KeyedVectors
import numpy as np
import torch
from torch import nn

from collections import OrderedDict
import os
import json
import logging
import math



def wordvec2npy(wv_file, out_path):
    """convert word2vec format file to numpy matrix npy_file and word2id json file 
        notes: word2vec file should ensure the first line is (vocab_size, emb_size)

    Args:
        wv_file (str): word2vec format file
        out_path (str): output path of numpy matrix npy_file and word2id json file
    """
    wv_model = KeyedVectors.load_word2vec_format(wv_file, binary=False)
    vocab_size = len(wv_model.vocab)
    emb_size = len(wv_model[list(wv_model.vocab.keys())[0]])
    word_emb_npy = np.zeros((vocab_size, emb_size))
    word2id = OrderedDict()
    idx = 0
    for w in wv_model.vocab.keys():
        word_emb_npy[idx] = wv_model[w]
        word2id[w] = idx
        idx += 1
    np.save(os.path.join(out_path, f'w2v.{vocab_size//1000}k.{emb_size}d_mat.npy'), word_emb_npy)
    with open(os.path.join(out_path, f'w2v.{vocab_size//1000}k.{emb_size}d_word2id.json'), 'w', encoding='utf-8') as f:
        json.dump(word2id, f, ensure_ascii=False)

def load_wordvec(wv_file, binary=False):
    """load word2vec format file, and return word2id dict and numpy mat of wordvec
        notes: word2vec file should ensure the first line is (vocab_size, emb_size)

    Args:
        wv_file (str): word2vec format file.
        binary (bool, optional): whether word2vec file is binary saced.

    Returns:
        word2id (dict): word2id dict
        word_emb_npy (numpy.array): numpy mat of wordvec
    """
    if binary:
        wv_model = KeyedVectors.load(wv_file) # load saved by wv.save()
    else:
        wv_model = KeyedVectors.load_word2vec_format(wv_file, binary=False)
    vocab_size = len(wv_model.vocab)
    emb_size = len(wv_model[list(wv_model.vocab.keys())[0]])
    word_emb_npy = np.zeros((vocab_size, emb_size))
    word2id = OrderedDict()
    idx = 0
    for w in wv_model.vocab.keys():
        word_emb_npy[idx] = wv_model[w]
        word2id[w] = idx
        idx += 1
    return word2id, word_emb_npy


def load_vocab_npy(vocab_file, w2v_npy_file):
    """load vocab and word embedding from vocab json format file
        and embedding npy format file

    Args:
        vocab_file (str): vocab json format file
        w2v_npy_file (str): embedding npy format file

    Returns:
        word2id (dict): word2id dict
        word_emb_npy (numpy.array): numpy mat of wordvec
    """
    word2id = json.load(open(vocab_file, encoding='utf-8'))
    word_emb_npy = np.load(w2v_npy_file)

    return word2id, word_emb_npy


def construct_embedding_from_numpy(word2id, word_size=50, word2vec=None, finetune=False):
    """construct embedding from numpy word2vec

    Args:
        word2id (dict): dictionary of word->idx mapping.
        word_size (int, optional): size of word embedding. Defaults to 50.
        word2vec (numpy.ndarray, optional): pretrained word2vec numpy. Defaults to None.
        finetune (bool, optional): whether finetune on word embedding.

    Returns:
        word2id (dict): updated dictionary of word->idx mapping.
        word_embedding (torch.nn.Embedding): torch embedding.
    """
    # load word vectors
    num_word = len(word2id)
    if word2vec is None:
        word_size = word_size
    else:
        word_size = word2vec.shape[-1]
    if word2vec is not None:
        try:
            word2vec = torch.from_numpy(word2vec)
        except TypeError as e:
            logging.info(e)
    # word vocab
    if not '[CLS]' in word2id:
        word2id['[CLS]'] = len(word2id)
        num_word += 1
        if word2vec is not None:
            bound = 1 / math.sqrt(word_size)
            cls_vec = nn.init.uniform_(torch.empty(1, word_size), -bound, bound) 
            word2vec = torch.cat([word2vec, cls_vec], dim=0)
    if not '[SEP]' in word2id:
        word2id['[SEP]'] = len(word2id)
        num_word += 1
        if word2vec is not None:
            bound = 1 / math.sqrt(word_size)
            sep_vec = nn.init.uniform_(torch.empty(1, word_size), -bound, bound) 
            word2vec = torch.cat([word2vec, sep_vec], dim=0)
    if not '[UNK]' in word2id:
        word2id['[UNK]'] = len(word2id)
        num_word += 1
        if word2vec is not None:
            bound = 1 / math.sqrt(word_size)
            unk_vec = nn.init.uniform_(torch.empty(1, word_size), -bound, bound) 
            word2vec = torch.cat([word2vec, unk_vec], dim=0)
    if not '[PAD]' in word2id:
        word2id['[PAD]'] = len(word2id)
        num_word += 1
        if word2vec is not None:
            blk_vec = torch.zeros(1, word_size)
            word2vec = torch.cat([word2vec, blk_vec], dim=0)
    # word embedding
    word_embedding = nn.Embedding(num_word, word_size)
    if word2vec is not None:
        if 'pad' in word2id:
            word2vec[word2id['[PAD]']] = word2vec[word2id['pad']]
        if 'cls' in word2id:
            word2vec[word2id['[CLS]']] = word2vec[word2id['cls']]
        if 'sep' in word2id:
            word2vec[word2id['[SEP]']] = word2vec[word2id['sep']]
        if 'unk' in word2id:
            word2vec[word2id['[UNK]']] = word2vec[word2id['unk']]
        logging.info("Initializing word embedding with word2vec.")
        word_embedding.weight.data.copy_(word2vec)
    word_embedding.weight.requires_grad = finetune

    return word2id, word_embedding
