"""
 Author: liujian
 Date: 2020-10-26 17:26:20
 Last Modified by: liujian
 Last Modified time: 2020-10-26 17:26:20
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..encoder import BERTEncoder
from ..decoder import CRF
from ...utils.entity_extract import *


class BILSTM_CRF(nn.Module):
    def __init__(self, sequence_encoder, tag2id, compress_seq=True, use_lstm=False, use_crf=True, tagscheme='bmoes', batch_first=True):
        """
        Args:
            sequence_encoder (nn.Module): encoder of sequence 
            tag2id (dict): map from tag to id
            compress_seq (bool, optional): whether compress sequence for lstm. Defaults to True.
            use_lstm (bool, optional): whether add lstm layer. Defaults to False.
            use_crf (bool, optional): whether add crf layer. Defaults to True.
            batch_first (bool, optional): whether fisrt dim is batch. Defaults to True.
        """
        
        super(BILSTM_CRF, self).__init__()
        self.batch_first = batch_first
        self.compress_seq = compress_seq
        self.tagscheme = tagscheme
        self.sequence_encoder = sequence_encoder
        self.mlp = nn.Linear(sequence_encoder.hidden_size, len(tag2id))
        self.softmax = nn.Softmax(dim=-1)
        self.tag2id = tag2id
        self.id2tag = {}
        for tag, tid in tag2id.items():
            self.id2tag[tid] = tag

        if use_lstm:
            self.bilstm = nn.LSTM(input_size=sequence_encoder.hidden_size, 
                                hidden_size=sequence_encoder.hidden_size, 
                                num_layers=1, 
                                bidirectional=True, 
                                batch_first=batch_first)
        else:
            self.bilstm = None
        if use_crf:
            self.crf = CRF(len(tag2id), batch_first=batch_first)
        else:
            self.crf = None


    def infer(self, text):
        """model inference
        Args:
            text (str or list): tokens list or sentence string
        
        Returns:
            pos_attr_entities (list[tuple]): list of (pos, entity_attr, entity)
        """
        self.eval()
        items = self.sequence_encoder.tokenize(text)
        logits = self.forward(*items)
        if self.crf is not None:
            preds = self.crf.decode(logits, mask=items[-1])[0]
        else:
            probs = self.softmax(logits)
            preds = probs.argmax(dim=-1).squeeze().numpy()[:items[-1].sum()]
        if 'bert' in self.sequence_encoder.__class__.__name__.lower():
            tags = [self.id2tag[tid] for tid in preds[1:-1]] # 包含'[CLS]'和'[SEP]'
        else:
            tags = [self.id2tag[tid] for tid in preds]
        if isinstance(text, str):
            text = self.sequence_encoder.tokenizer.tokenize(text)
        pos_attr_entities = eval(f'extract_kvpairs_in_{self.tagscheme}')(tags, text)
            
        return text, pos_attr_entities


    def forward(self, *args):
        if not hasattr(self, '_flattened'):
            if self.bilstm is not None:
                self.bilstm.flatten_parameters()
                setattr(self, '_flattened', True)
        rep = self.sequence_encoder(*args) # B, S, D
        if self.bilstm is not None:
            if self.compress_seq:
                att_mask = args[-1]
                seqs_length = att_mask.sum(dim=-1).detach().cpu()
                seqs_rep_packed = pack_padded_sequence(rep, seqs_length, batch_first=self.batch_first)
                seqs_hiddens_packed, _ = self.bilstm(seqs_rep_packed)
                seqs_hiddens, _ = pad_packed_sequence(seqs_hiddens_packed, batch_first=self.batch_first) # B, S, D
            else:
                seqs_hiddens, _ = self.bilstm(rep)
            # seqs_hiddens = nn.functional.dropout(seqs_hiddens, 0.2)
            seqs_hiddens = torch.add(*seqs_hiddens.chunk(2, dim=-1))
        else:
            # seqs_hiddens = torch.cat([rep, rep], dim=-1) # keep the same dimension with bilstm hiddens
            seqs_hiddens = rep

        logits_seq = self.mlp(seqs_hiddens) # B, L, V

        return logits_seq