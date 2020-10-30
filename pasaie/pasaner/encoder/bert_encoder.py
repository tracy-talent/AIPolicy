"""
 Author: liujian
 Date: 2020-10-26 17:54:15
 Last Modified by: liujian
 Last Modified time: 2020-10-26 17:54:15
"""

import logging
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BERTEncoder(nn.Module):
    def __init__(self, max_length, pretrain_path, blank_padding=True):
        """
        Args:
            pretrain_path: path of pretrain model
        """
        super(BERTEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.hidden_size = self.bert.config.hidden_size
        self.max_length = max_length
        self.blank_padding = blank_padding

    def forward(self, seqs, att_mask):
        """
        Args:
            seqs: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        seq_out, _ = self.bert(seqs, attention_mask=att_mask)
        return seq_out
    
    def tokenize(self, text):
        if isinstance(text, list) or isinstance(text, tuple):
            sentence = text
            is_token = True
        else:
            sentence = text
            is_token = False
        
        if is_token:
            # tokens = self.tokenizer.tokenize(''.join(sentence))
            tokens = sentence
        else:
            tokens = self.tokenizer.tokenize(sentence)
        
        re_tokens = ['[CLS]'] + tokens + ['[SEP]']
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avail_len = torch.tensor([len(indexed_tokens)])

        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)

        # attention mask
        att_mask = torch.zeros(indexed_tokens.size(), dtype=torch.uint8) # (1, L)
        att_mask[0, :avail_len] = 1

        return indexed_tokens, att_mask  # ensure the first and last is indexed_tokens and att_mask