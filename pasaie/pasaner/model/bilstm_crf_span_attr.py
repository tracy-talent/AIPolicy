"""
 Author: liujian
 Date: 2020-10-26 17:48:40
 Last Modified by: liujian
 Last Modified time: 2020-10-26 17:48:40
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..encoder import BERTEncoder
from ..decoder import CRF
from ...utils.entity_extract import *


class BILSTM_CRF_Span_Attr(nn.Module):
    def __init__(self, sequence_encoder, span2id, attr2id, compress_seq=False, share_lstm=False, span_use_lstm=True, attr_use_lstm=False, span_use_crf=True, attr_use_crf=False, tagscheme='bmoes', batch_first=True):
        """
        Args:
            sequence_encoder (nn.Module): encoder of sequence
            span2id (dict): map from span(et. B, I, O) to id
            attr2id (dict): map from attr(et. PER, LOC, ORG) to id
            compress_seq (bool, optional): whether compress sequence for lstm. Defaults to True.
            share_lstm (bool, optional): whether make span and attr share the same lstm after encoder. Defaults to False.
            span_use_lstm (bool, optional): whether add span lstm layer. Defaults to True.
            span_use_lstm (bool, optional): whether add attr lstm layer. Defaults to False.
            span_use_crf (bool, optional): whether add span crf layer. Defaults to True.
            attr_use_crf (bool, optional): whether add attr crf layer. Defaults to False.
            batch_first (bool, optional): whether fisrt dim is batch. Defaults to True.
        """
        
        super(BILSTM_CRF_Span_Attr, self).__init__()
        self.batch_first = batch_first
        self.compress_seq = compress_seq
        self.tagscheme = tagscheme
        self.sequence_encoder = sequence_encoder
        self.mlp_span = nn.Linear(sequence_encoder.hidden_size, len(span2id))
        self.mlp_attr = nn.Linear(sequence_encoder.hidden_size, len(attr2id))
        self.softmax = nn.Softmax(dim=-1)
        self.span2id = span2id
        self.id2span = {}
        for span, sid in span2id.items():
            self.id2span[sid] = span
        self.attr2id = attr2id
        self.id2attr = {}
        for attr, aid in attr2id.items():
            self.id2attr[aid] = attr

        self.span_bilstm = None
        self.attr_bilstm = None
        self.share_bilstm = None
        if share_lstm:
            self.share_bilstm = nn.LSTM(input_size=sequence_encoder.hidden_size, 
                                hidden_size=sequence_encoder.hidden_size, 
                                num_layers=1, 
                                bidirectional=True, 
                                batch_first=batch_first)
        else:
            if span_use_lstm:
                self.span_bilstm = nn.LSTM(input_size=sequence_encoder.hidden_size, 
                                    hidden_size=sequence_encoder.hidden_size, 
                                    num_layers=1, 
                                    bidirectional=True, 
                                    batch_first=batch_first)
            if attr_use_lstm:
                self.attr_bilstm = nn.LSTM(input_size=sequence_encoder.hidden_size, 
                                    hidden_size=sequence_encoder.hidden_size, 
                                    num_layers=1, 
                                    bidirectional=True, 
                                    batch_first=batch_first)
        if span_use_crf:
            self.crf_span = CRF(len(span2id), batch_first=batch_first)
        else:
            self.crf_span = None
        if attr_use_crf:
            self.crf_attr = CRF(len(attr2id), batch_first=batch_first)
        else:
            self.crf_attr = None
        
    
    def infer(self, text):
        """model inference
        Args:
            text (str or list): tokens list or sentence string

        Returns:
            pos_attr_entities (list[tuple]): list of (pos, entity_attr, entity)
        """
        self.eval()
        items = self.sequence_encoder.tokenize(text)
        logits_span, logits_attr = self.forward(*items)
        if self.crf_span is not None:
            preds_span = self.crf_span.decode(logits_span, mask=items[-1])[0]
        else:
            probs_span = self.softmax(logits_span)
            preds_span = probs_span.argmax(dim=-1).squeeze().numpy()[:items[-1].sum()]
        if self.crf_attr is not None:
            preds_attr = self.crf_attr.decode(logits_attr, mask=items[-1])[0]
        else:
            probs_attr = self.softmax(logits_attr)
            preds_attr = probs_attr.argmax(dim=-1).squeeze().numpy()[:items[-1].sum()]

        if 'bert' in self.sequence_encoder.__class__.__name__.lower():
            spans = [self.id2span[sid] for sid in preds_span[1:-1]] # 包含'[CLS]'和'[SEP]'
            attrs = [self.id2attr[aid] for aid in preds_attr[1:-1]]
        else:
            spans = [self.id2span[sid] for sid in preds_span]
            attrs = [self.id2attr[aid] for aid in preds_attr]
        tags = [span + '-' + attr if span != 'O' else 'O' for span, attr in zip(spans, attrs)]
        if isinstance(text, str):
            text = self.sequence_encoder.tokenizer.tokenize(text)
        pos_attr_entities = eval(f'extract_kvpairs_in_{self.tagscheme}')(tags, text)

        return text, pos_attr_entities


    def forward(self, span_labels, *args):
        if not hasattr(self, '_flattened'):
            if self.share_bilstm is not None:
                self.share_bilstm.flatten_parameters()
            if self.span_bilstm is not None:
                self.span_bilstm.flatten_parameters()
            if self.attr_bilstm is not None:
                self.attr_bilstm.flatten_parameters()
            setattr(self, '_flattened', True)
        rep = self.sequence_encoder(*args) # B, S, D
        # span_seqs_hiddens = torch.cat([rep, rep], dim=-1) # keep the same dimension with bilstm hiddens
        # attr_seqs_hiddens = torch.cat([rep, rep], dim=-1) # keep the same dimension with bilstm hiddens
        span_seqs_hiddens = rep
        attr_seqs_hiddens = rep
        if self.share_bilstm is not None:
            if self.compress_seq:
                att_mask = args[-1]
                seqs_length = att_mask.sum(dim=-1).detach().cpu()
                seqs_rep_packed = pack_padded_sequence(rep, seqs_length, batch_first=self.batch_first)
                seqs_hiddens_packed, _ = self.share_bilstm(seqs_rep_packed)
                span_seqs_hiddens, _ = pad_packed_sequence(seqs_hiddens_packed, batch_first=self.batch_first) # B, S, D
            else:
                span_seqs_hiddens, _ = self.share_bilstm(rep)
            span_seqs_hiddens = torch.add(*span_seqs_hiddens.chunk(2, dim=-1))
            attr_seqs_hiddens = span_seqs_hiddens
        else:
            if self.span_bilstm is not None:
                if self.compress_seq:
                    att_mask = args[-1]
                    seqs_length = att_mask.sum(dim=-1).detach().cpu()
                    seqs_rep_packed = pack_padded_sequence(rep, seqs_length, batch_first=self.batch_first)
                    seqs_hiddens_packed, _ = self.span_bilstm(seqs_rep_packed)
                    span_seqs_hiddens, _ = pad_packed_sequence(seqs_hiddens_packed, batch_first=self.batch_first) # B, S, D
                else:
                    span_seqs_hiddens, _ = self.span_bilstm(rep)
                span_seqs_hiddens = torch.add(*span_seqs_hiddens.chunk(2, dim=-1))
                if self.attr_bilstm is not None:
                    if self.compress_seq:
                        seqs_hiddens_packed, _ = self.attr_bilstm(seqs_rep_packed)
                        attr_seqs_hiddens, _ = pad_packed_sequence(seqs_hiddens_packed, batch_first=self.batch_first) # B, S, D
                    else:
                        attr_seqs_hiddens, _ = self.attr_bilstm(rep)
                    attr_seqs_hiddens = torch.add(*attr_seqs_hiddens.chunk(2, dim=-1))
            elif self.attr_bilstm is not None:
                if self.compress_seq:
                    att_mask = args[-1]
                    seqs_length = att_mask.sum(dim=-1).detach().cpu()
                    seqs_rep_packed = pack_padded_sequence(rep, seqs_length, batch_first=self.batch_first)
                    seqs_hiddens_packed, _ = self.attr_bilstm(seqs_rep_packed)
                    attr_seqs_hiddens, _ = pad_packed_sequence(seqs_hiddens_packed, batch_first=self.batch_first) # B, S ,D
                else:
                    attr_seqs_hiddens, _ = self.attr_bilstm(rep)
                attr_seqs_hiddens = torch.add(*attr_seqs_hiddens.chunk(2, dim=-1))

        if span_labels is not None:
            span_bid, span_eid = self.span2id['B'], self.span2id['E']
            spos = -1
            for i in range(span_labels.size(0)):
                for j in range(span_labels.size(1)):
                    if span_labels[i][j].item() == span_bid:
                        spos = j
                    elif span_labels[i][j].item() == span_eid:
                        attr_seqs_hiddens[i][j] = torch.mean(attr_seqs_hiddens[i][spos:j+1], dim=0)

        logits_span = self.mlp_span(span_seqs_hiddens) # B, S, V
        logits_attr = self.mlp_attr(attr_seqs_hiddens) # B, S, V
        
        return logits_span, logits_attr