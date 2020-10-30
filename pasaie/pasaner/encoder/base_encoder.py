"""
 Author: liujian 
 Date: 2020-10-25 14:29:08 
 Last Modified by: liujian 
 Last Modified time: 2020-10-25 14:29:08 
"""

import math, logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ...tokenization import WordTokenizer

class BaseEncoder(nn.Module):

    def __init__(self, 
                 token2id, 
                 max_length=256, 
                 hidden_size=230, 
                 word_size=50,
                 word2vec=None,
                 blank_padding=True):
        """
        Args:
            token2id: dictionary of token->idx mapping
            max_length: max length of sentence, used for postion embedding
            hidden_size: hidden size
            word_size: size of word embedding
            blank_padding: padding for CNN
            word2vec: pretrained word2vec numpy
        """
        # Hyperparameters
        super().__init__()

        self.token2id = token2id
        self.max_length = max_length
        self.num_token = len(token2id)

        if word2vec is None:
            self.word_size = word_size
        else:
            self.word_size = word2vec.shape[-1]
            
        self.hidden_size = hidden_size
        self.input_size = self.word_size
        self.blank_padding = blank_padding

        if not '[UNK]' in self.token2id:
            self.token2id['[UNK]'] = len(self.token2id)
            self.num_token += 1
        if not '[PAD]' in self.token2id:
            self.token2id['[PAD]'] = len(self.token2id)
            self.num_token += 1

        # Word embedding
        self.word_embedding = nn.Embedding(self.num_token, self.word_size)
        if word2vec is not None:
            logging.info("Initializing word embedding with word2vec.")
            word2vec = torch.from_numpy(word2vec)
            if self.num_token == len(word2vec) + 2:            
                unk = torch.randn(1, self.word_size) / math.sqrt(self.word_size)
                blk = torch.zeros(1, self.word_size)
                self.word_embedding.weight.data.copy_(torch.cat([word2vec, unk, blk], 0))
            else:
                self.word_embedding.weight.data.copy_(word2vec)

        # Position Embedding
        self.tokenizer = WordTokenizer(vocab=self.token2id, unk_token="[UNK]")

    def forward(self, seqs, att_mask):
        """
        Args:
            seqs: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, L, H), representations for sequences
        """
        # Check size of tensors
        inputs_embed = self.word_embedding(seqs) # (B, L, EMBED)
        return inputs_embed
    
    def tokenize(self, text):
        """
        Args:
            item: input instance, including sentence, entity positions, etc.
            is_token: if is_token == True, sentence becomes an array of token
        Return:
            index number of tokens and positions             
        """
        if isinstance(text, list) or isinstance(text, tuple):
            sentence = text
            is_token = True
        else:
            sentence = text
            is_token = False

        # Sentence -> token
        if not is_token:
            tokens = self.tokenizer.tokenize(sentence)        
        else:
            tokens = sentence

        avail_len = torch.tensor([len(tokens)]) # 序列实际长度

        # Token -> index
        if self.blank_padding:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens, self.max_length, self.token2id['[PAD]'], self.token2id['[UNK]'])
        else:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens, unk_id = self.token2id['[UNK]'])
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)

        # attention mask
        att_mask = torch.zeros(indexed_tokens.size(), dtype=torch.uint8) # (1, L), crf reqiure mask dtype=bool or uint8
        att_mask[0, :avail_len] = 1

        return indexed_tokens, att_mask
