"""
 Author: liujian
 Date: 2020-10-26 17:54:15
 Last Modified by: liujian
 Last Modified time: 2020-10-26 17:54:15
"""

import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import BertModel, AlbertModel, BertTokenizer


class BERT_BILSTM_Encoder(nn.Module):
    def __init__(self, max_length, pretrain_path, bert_name, use_lstm=False, compress_seq=False, blank_padding=True):
        """
        Args:
            max_length (int): max length of sequence
            pretrain_path (str): path of pretrain model
            use_lstm (bool, optional): whether add lstm layer. Defaults to False.
            compress_seq (bool, optional): whether compress sequence for lstm. Defaults to True.
            bert_name (str): model name of bert series model, such as bert, roberta, xlnet, albert
            blank_padding (bool, optional): whether pad sequence to max length. Defaults to True.
        """
        super(BERT_BILSTM_Encoder, self).__init__()

        self.bert_name = bert_name
        if 'albert' in bert_name:
            self.bert = AlbertModel.from_pretrained(pretrain_path) # clue
            self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        if 'roberta' in bert_name:
            # self.bert = AutoModelForMaskedLM.from_pretrained(pretrain_path, output_hidden_states=True) # hfl
            # self.tokenizer = AutoTokenizer.from_pretrained(pretrain_path) # hfl
            self.bert = BertModel.from_pretrained(pretrain_path) # clue
            self.tokenizer = BertTokenizer.from_pretrained(pretrain_path) # clue
        elif 'bert' in bert_name:
            # self.bert = AutoModelForMaskedLM.from_pretrained(pretrain_path, output_hidden_states=True)
            self.bert = BertModel.from_pretrained(pretrain_path)
            self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        # add missed tokens in vocab.txt
        num_added_tokens = self.tokenizer.add_tokens(['“', '”', '—'])
        print(f"we have added {num_added_tokens} tokens ['“', '”', '—']")
        self.bert.resize_token_embeddings(len(self.tokenizer))

        self.hidden_size = self.bert.config.hidden_size
        self.max_length = max_length
        self.blank_padding = blank_padding

        # bilstm 
        if use_lstm:
            self.bilstm = nn.LSTM(input_size=self.hidden_size, 
                                hidden_size=self.hidden_size, 
                                num_layers=1, 
                                bidirectional=True, 
                                batch_first=True)
        else:
            self.bilstm = None
        self.compress_seq = compress_seq

    def forward(self, seqs, att_mask):
        """
        Args:
            seqs: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        if not hasattr(self, '_flattened'):
            if self.bisltm is not None:
                self.bilstm.flatten_parameters()
            setattr(self, '_flattened', True)
        if 'roberta' in self.bert_name:
            # seq_out = self.bert(seqs, attention_mask=att_mask)[1][1] # hfl roberta
            seq_out, _ = self.bert(seqs, attention_mask=att_mask) # clue-roberta
        else:
            seq_out, _ = self.bert(seqs, attention_mask=att_mask)

        if self.bilstm is not None:
            if self.compress_seq:
                seqs_length = att_mask.sum(dim=-1).detach().cpu()
                seqs_rep_packed = pack_padded_sequence(seq_out, seqs_length, batch_first=True)
                seqs_hiddens_packed, _ = self.bilstm(seqs_rep_packed)
                seqs_hiddens, _ = pad_packed_sequence(seqs_hiddens_packed, batch_first=True) # B, S, D
            else:
                seqs_hiddens, _ = self.bilstm(seq_out)
            # seqs_hiddens = nn.functional.dropout(seqs_hiddens, 0.2)
            seq_out = torch.add(*seqs_hiddens.chunk(2, dim=-1))
            
        return seq_out
    
    def tokenize(self, *items): # items = (tokens, spans, [attrs, optional])
        """
        Args:
            items: (tokens, tags) or (tokens, spans, atrrs) or (sentence)
        Returns:
            indexed_tokens (torch.tensor): tokenizer encode ids of tokens, (1, L)
            att_mask (torch.tensor): token mask ids, (1, L)
        """
        if isinstance(items[0], list) or isinstance(items[0], tuple):
            sentence = items[0]
            is_token = True
        else:
            sentence = items[0]
            is_token = False
        
        if is_token:
            items[0].insert(0, '[CLS]')
            items[0].append('[SEP]')
            if len(items) > 1:
                items[1].insert(0, 'O')
                items[1].append('O')
            if len(items) > 2:
                items[2].insert(0, 'null')
                items[2].append('null')
            tokens = items[0]
        else:
            tokens = ['[CLS]'] + self.tokenizer.tokenize(sentence) + ['SEP']
        
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        avail_len = torch.tensor([len(indexed_tokens)])

        if self.blank_padding:
            if len(indexed_tokens) <= self.max_length:
                while len(indexed_tokens) < self.max_length:
                    indexed_tokens.append(0)
            else:
                indexed_tokens[self.max_length - 1] = indexed_tokens[-1]
                indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)

        # attention mask
        att_mask = torch.zeros(indexed_tokens.size(), dtype=torch.uint8) # (1, L)
        att_mask[0, :avail_len] = 1

        return indexed_tokens, att_mask  # ensure the first and last is indexed_tokens and att_mask