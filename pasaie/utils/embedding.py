from gensim.models import KeyedVectors
import numpy as np

from collections import OrderedDict
import os
import json

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

def load_wordvec(wv_file):
    """load word2vec format file, and return word2id dict and numpy mat of wordvec
        notes: word2vec file should ensure the first line is (vocab_size, emb_size)

    Args:
        wv_file (str): word2vec format file

    Returns:
        word2id (dict): word2id dict
        word_emb_npy (numpy.array): numpy mat of wordvec
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