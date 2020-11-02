"""
 Author: liujian 
 Date: 2020-10-25 12:35:36 
 Last Modified by: liujian 
 Last Modified time: 2020-10-25 12:35:36 
"""

import math, logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ...tokenization import WordTokenizer
from ...tokenization import JiebaTokenizer


class BaseWLFEncoder(nn.Module):
    """word level feature(wlf) + char feature
    """
    def __init__(self, 
                 char2id, 
                 word2id,
                 max_length=256, 
                 hidden_size=230, 
                 char_size=50,
                 word_size=50,
                 char2vec=None,
                 word2vec=None,
                 custom_dict=None,
                 blank_padding=True):
        """
        Args:
            char2id (dict): dictionary of char->idx mapping
            word2id (dict): dictionary of word->idx mapping
            max_length (int): max length of sentence, used for postion embedding
            hidden_size (int): hidden size
            char_size (int): size of char embedding
            word_size (int): size of word embedding
            char2vec (numpy.array): pretrained char2vec numpy
            word2vec (numpy.array): pretrained word2vec numpy
            custom_dict (dict): customized dictionary for word tokenizer
            blank_padding (bool): padding for indexed sequence
        """
        # Hyperparameters
        super().__init__()

        self.char2id = char2id
        self.word2id = word2id
        self.max_length = max_length
        self.num_char = len(char2id)
        self.num_word = len(word2id)

        if char2vec is None:
            self.char_size = char_size
        else:
            self.char_size = char2vec.shape[-1]
        if word2vec is None:
            self.word_size = word_size
        else:
            self.word_size = word2vec.shape[-1]
            
        self.hidden_size = hidden_size
        self.input_size = self.char_size + self.word_size
        self.blank_padding = blank_padding

        # char vocab
        if not '[UNK]' in self.char2id:
            self.char2id['[UNK]'] = len(self.char2id)
            self.num_char += 1
        if not '[PAD]' in self.char2id:
            self.char2id['[PAD]'] = len(self.char2id)
            self.num_char += 1
        # Char embedding
        self.char_embedding = nn.Embedding(self.num_char, self.char_size)
        if char2vec is not None:
            logging.info("Initializing char embedding with char2vec.")
            char2vec = torch.from_numpy(char2vec)
            if self.num_char == len(char2vec) + 2:            
                unk = torch.randn(1, self.char_size) / math.sqrt(self.char_size)
                blk = torch.zeros(1, self.char_size)
                self.char_embedding.weight.data.copy_(torch.cat([char2vec, unk, blk], 0))
            else:
                self.char_embedding.weight.data.copy_(char2vec)

        if self.char2id is not self.word2id:
            # word vocab
            if not '[UNK]' in self.word2id:
                self.word2id['[UNK]'] = len(self.word2id)
                self.num_word += 1
            if not '[PAD]' in self.word2id:
                self.word2id['[PAD]'] = len(self.word2id)
                self.num_word += 1
            # Word embedding
            self.word_embedding = nn.Embedding(self.num_word, self.word_size)
            if word2vec is not None:
                logging.info("Initializing word embedding with word2vec.")
                word2vec = torch.from_numpy(word2vec)
                if self.num_word == len(word2vec) + 2:            
                    unk = torch.randn(1, self.word_size) / math.sqrt(self.word_size)
                    blk = torch.zeros(1, self.word_size)
                    self.word_embedding.weight.data.copy_(torch.cat([word2vec, unk, blk], 0))
                else:
                    self.word_embedding.weight.data.copy_(word2vec)
        else:
            self.word_embedding = self.char_embedding

        # tokenizer
        self.tokenizer = WordTokenizer(vocab=self.char2id, unk_token="[UNK]")
        self.word_tokenizer = JiebaTokenizer(vocab=self.word2id, unk_token="[UNK]", custom_dict=custom_dict)

    def forward(self, seqs_char, seqs_word, att_mask):
        """
        Args:
            seqs_char: (B, L), index of chars
            seqs_word: (B, L), index of words
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, L, H), representations for sequences
        """
        # Check size of tensors
        inputs_embed = torch.cat([
            self.char_embedding(seqs_char),
            self.word_embedding(seqs_word)
        ], dim=-1) # (B, L, EMBED)
        return inputs_embed
    
    def tokenize(self, *items):
        """
        Args:
            items: (tokens, tags) or (tokens, spans, atrrs) or (sentence)
        Return:
            index number of tokens and positions             
        """
        if isinstance(items[0], list) or isinstance(items[0], tuple):
            sentence = items[0]
            is_token = True
        else:
            sentence = items[0]
            is_token = False

        # Sentence -> token
        if not is_token:
            tokens = self.tokenizer.tokenize(sentence)        
        else:
            tokens = sentence

        avail_len = torch.tensor([len(tokens)]) # 序列实际长度
        words = self.word_tokenizer.tokenize(''.join(tokens))
        token2word = [0] * avail_len
        wpos, wlen = 0, 0
        for i in range(avail_len):
            if i >= wlen + len(words[wpos]):
                wlen += len(words[wpos])
                wpos += 1
            token2word[i] = wpos

        # Token -> index
        if self.blank_padding:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens, self.max_length, self.char2id['[PAD]'], self.char2id['[UNK]'])
            indexed_words = self.word_tokenizer.convert_tokens_to_ids(words, unk_id=self.word2id['[UNK]'])
            indexed_token2word = [self.char2id['[PAD]']] * len(indexed_tokens)
            for i in range(avail_len):
                indexed_token2word[i] = indexed_words[token2word[i]]
        else:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens, unk_id = self.char2id['[UNK]'])
            indexed_words = self.word_tokenizer.convert_tokens_to_ids(words, unk_id=self.word2id['[UNK]'])
            indexed_token2word = [self.char2id['[PAD]']] * len(indexed_tokens)
            for i in range(avail_len):
                indexed_token2word[i] = indexed_words[token2word[i]]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)
        indexed_token2word = torch.tensor(indexed_token2word).long().unsqueeze(0) # (1, L)

        # attention mask
        att_mask = torch.zeros(indexed_tokens.size(), dtype=torch.uint8) # (1, L)
        att_mask[0, :avail_len] = 1

        return indexed_tokens, indexed_token2word, att_mask
