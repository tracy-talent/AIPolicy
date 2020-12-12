import logging
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from transformers import BertModel, BertTokenizer


class BERTEncoder(nn.Module):
    def __init__(self, max_length, pretrain_path, blank_padding=True):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model
        """
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        # add possible missed tokens in vocab.txt
        num_added_tokens = self.tokenizer.add_tokens(['“', '”', '—'])
        logging.info(f"we have added {num_added_tokens} tokens ['“', '”', '—']")
        self.bert.resize_token_embeddings(len(self.tokenizer))

    def forward(self, token, att_mask):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        seq_out, sent_out = self.bert(token, attention_mask=att_mask)
        return seq_out

    def tokenize(self, item):
        """
        Args:
            item: data instance containing 'text' / 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        # Sentence -> token
        if isinstance(item, tuple) or isinstance(item, list):
            sentence = item[0]
        else:
            sentence = item

        re_tokens = self.tokenizer.tokenize(sentence)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        # Padding
        if self.blank_padding:
            if len(indexed_tokens) <= self.max_length:
                while len(indexed_tokens) < self.max_length:
                    indexed_tokens.append(0)  # 0 is id for [PAD]
            else:
                indexed_tokens[self.max_length - 1] = indexed_tokens[-1]
                indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask
