"""
 Author: liujian
 Date: 2021-01-03 11:13:24
 Last Modified by: liujian
 Last Modified time: 2021-01-03 11:13:24
"""

from functools import total_ordering
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..decoder import CRF
from ...utils.entity_extract import *


class Base_BILSTM_CRF_Span_Attr(nn.Module):
    def __init__(self, sequence_encoder, span2id, attr2id, compress_seq=False, share_lstm=False, span_use_lstm=True, attr_use_lstm=False, span_use_crf=True, tagscheme='bmoes', batch_first=True, dropout_rate=0.3):
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
            batch_first (bool, optional): whether fisrt dim is batch. Defaults to True.
            dropout_rate (float, optional): dropout rate. Defaults to 0.3.
        """
        
        super(Base_BILSTM_CRF_Span_Attr, self).__init__()
        self.batch_first = batch_first
        self.compress_seq = compress_seq
        self.tagscheme = tagscheme
        self.sequence_encoder = sequence_encoder
        self.mlp_span = nn.Linear(sequence_encoder.hidden_size, len(span2id))
        self.mlp_attr_start = nn.Linear(sequence_encoder.hidden_size, len(attr2id))
        self.mlp_attr_end = nn.Linear(sequence_encoder.hidden_size, len(attr2id))
        self.dropout = nn.Dropout(dropout_rate)
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

    
    def infer(self, text):
        """model inference
        Args:
            text (str or list): tokens list or sentence string

        Returns:
            pos_attr_entities (list[tuple]): list of (pos, entity_attr, entity)
        """
        self.eval()
        negid = -1
        if 'null' in self.attr2id:
            negid = self.attr2id['null']
        if negid == -1:
            raise Exception("negative tag not in 'null'")
        seqs = self.sequence_encoder.tokenize(text)
        seq_len = seqs[-1].sum().item()
        logits_span, logits_attr_start, logits_attr_end = self.forward(*seqs)
        if self.crf_span is not None:
            preds_span = self.crf_span.decode(logits_span, mask=items[-1])[0]
        else:
            preds_span = probs_span.argmax(dim=-1).squeeze().numpy()[:seq_len]
        preds_attr_start = logits_attr_start[:,:seq_len,:].argmax(dim=-1).squeeze().detach().cpu().numpy()
        preds_attr_end = logits_attr_end[:,:seq_len,:].argmax(dim=-1).squeeze().detach().cpu().numpy()

        spos, tpos = 0, seq_len
        if 'bert' in self.sequence_encoder.__class__.__name__.lower():
            spos, tpos = 1, -1
        spans = [self.id2span[sid] for sid in preds_span[spos:tpos]] # 包含'[CLS]'和'[SEP]'
        attrs_start = [self.id2attr[aid] for aid in preds_attr_start[spos:tpos]]
        attrs_end = [self.id2attr[aid] for aid in preds_attr_end[spos:tpos]]
        tags = [span + '-' + attr if span != 'O' else 'O' for span, attr in zip(spans, attrs)]
        if isinstance(text, str):
            text = self.sequence_encoder.tokenizer.tokenize(text)
        # pos_attr_entities = eval(f'extract_kvpairs_in_{self.tagscheme}')(tags, text)
        pos_attr_entities = extract_kvpairs_by_start_end(attrs_start, attrs_end, text, self.id2attr[negid])

        return text, pos_attr_entities


    def forward(self, *args):
        if not hasattr(self, '_flattened'):
            if self.share_bilstm is not None:
                self.share_bilstm.flatten_parameters()
            if self.span_bilstm is not None:
                self.span_bilstm.flatten_parameters()
            if self.attr_bilstm is not None:
                self.attr_bilstm.flatten_parameters()
            setattr(self, '_flattened', True)
        rep = self.sequence_encoder(*args) # B, S, D
        self.encoder_output = rep
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
                span_seqs_hiddens, _ = pad_packed_sequence(seqs_hiddens_packed, batch_first=self.batch_first, total_length=att_mask.size(-1)) # B, S, D
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
                    span_seqs_hiddens, _ = pad_packed_sequence(seqs_hiddens_packed, batch_first=self.batch_first, total_length=att_mask.size(-1)) # B, S, D
                else:
                    span_seqs_hiddens, _ = self.span_bilstm(rep)
                span_seqs_hiddens = torch.add(*span_seqs_hiddens.chunk(2, dim=-1))
                if self.attr_bilstm is not None:
                    if self.compress_seq:
                        seqs_hiddens_packed, _ = self.attr_bilstm(seqs_rep_packed)
                        attr_seqs_hiddens, _ = pad_packed_sequence(seqs_hiddens_packed, batch_first=self.batch_first, total_length=att_mask.size(-1)) # B, S, D
                    else:
                        attr_seqs_hiddens, _ = self.attr_bilstm(rep)
                    attr_seqs_hiddens = torch.add(*attr_seqs_hiddens.chunk(2, dim=-1))
            elif self.attr_bilstm is not None:
                if self.compress_seq:
                    att_mask = args[-1]
                    seqs_length = att_mask.sum(dim=-1).detach().cpu()
                    seqs_rep_packed = pack_padded_sequence(rep, seqs_length, batch_first=self.batch_first)
                    seqs_hiddens_packed, _ = self.attr_bilstm(seqs_rep_packed)
                    attr_seqs_hiddens, _ = pad_packed_sequence(seqs_hiddens_packed, batch_first=self.batch_first, total_length=att_mask.size(-1)) # B, S ,D
                else:
                    attr_seqs_hiddens, _ = self.attr_bilstm(rep)
                attr_seqs_hiddens = torch.add(*attr_seqs_hiddens.chunk(2, dim=-1))
        
        return span_seqs_hiddens, attr_seqs_hiddens



class Base_BILSTM_CRF_Span_Attr_Three(nn.Module):
    def __init__(self, sequence_encoder, span2id, attr2id, compress_seq=False, share_lstm=False, span_use_lstm=True, attr_use_lstm=False, span_use_crf=True, tagscheme='bmoes', batch_first=True, dropout_rate=0.3):
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
            batch_first (bool, optional): whether fisrt dim is batch. Defaults to True.
            dropout_rate (float, optional): dropout rate. Defaults to 0.3.
        """
        
        super(Base_BILSTM_CRF_Span_Attr_Three, self).__init__()
        self.batch_first = batch_first
        self.compress_seq = compress_seq
        self.tagscheme = tagscheme
        self.sequence_encoder = sequence_encoder
        self.mlp_span = nn.Linear(sequence_encoder.hidden_size, len(span2id))
        self.mlp_attr_start = nn.Linear(sequence_encoder.hidden_size, len(attr2id))
        self.mlp_attr_end = nn.Linear(sequence_encoder.hidden_size, len(attr2id))
        self.dropout = nn.Dropout(dropout_rate)
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
                self.attr_bilstm_start = nn.LSTM(input_size=sequence_encoder.hidden_size, 
                                    hidden_size=sequence_encoder.hidden_size, 
                                    num_layers=1, 
                                    bidirectional=True, 
                                    batch_first=batch_first)
                self.attr_bilstm_end = nn.LSTM(input_size=sequence_encoder.hidden_size, 
                                    hidden_size=sequence_encoder.hidden_size, 
                                    num_layers=1, 
                                    bidirectional=True, 
                                    batch_first=batch_first)
        if span_use_crf:
            self.crf_span = CRF(len(span2id), batch_first=batch_first)
        else:
            self.crf_span = None

    
    def infer(self, text):
        """model inference
        Args:
            text (str or list): tokens list or sentence string

        Returns:
            pos_attr_entities (list[tuple]): list of (pos, entity_attr, entity)
        """
        self.eval()
        negid = -1
        if 'null' in self.attr2id:
            negid = self.attr2id['null']
        if negid == -1:
            raise Exception("negative tag not in 'null'")
        seqs = self.sequence_encoder.tokenize(text)
        seq_len = seqs[-1].sum().item()
        logits_span, logits_attr_start, logits_attr_end = self.forward(*seqs)
        if self.crf_span is not None:
            preds_span = self.crf_span.decode(logits_span, mask=items[-1])[0]
        else:
            preds_span = probs_span.argmax(dim=-1).squeeze().numpy()[:seq_len]
        preds_attr_start = logits_attr_start[:,:seq_len,:].argmax(dim=-1).squeeze().detach().cpu().numpy()
        preds_attr_end = logits_attr_end[:,:seq_len,:].argmax(dim=-1).squeeze().detach().cpu().numpy()

        spos, tpos = 0, seq_len
        if 'bert' in self.sequence_encoder.__class__.__name__.lower():
            spos, tpos = 1, -1
        spans = [self.id2span[sid] for sid in preds_span[spos:tpos]] # 包含'[CLS]'和'[SEP]'
        attrs_start = [self.id2attr[aid] for aid in preds_attr_start[spos:tpos]]
        attrs_end = [self.id2attr[aid] for aid in preds_attr_end[spos:tpos]]
        tags = [span + '-' + attr if span != 'O' else 'O' for span, attr in zip(spans, attrs)]
        if isinstance(text, str):
            text = self.sequence_encoder.tokenizer.tokenize(text)
        # pos_attr_entities = eval(f'extract_kvpairs_in_{self.tagscheme}')(tags, text)
        pos_attr_entities = extract_kvpairs_by_start_end(attrs_start, attrs_end, text, self.id2attr[negid])

        return text, pos_attr_entities


    def forward(self, *args):
        if not hasattr(self, '_flattened'):
            if self.share_bilstm is not None:
                self.share_bilstm.flatten_parameters()
            if self.span_bilstm is not None:
                self.span_bilstm.flatten_parameters()
            if self.attr_bilstm_start is not None:
                self.attr_bilstm_start.flatten_parameters()
            if self.attr_bilstm_end is not None:
                self.attr_bilstm_end.flatten_parameters()
            setattr(self, '_flattened', True)
        rep = self.sequence_encoder(*args) # B, S, D
        self.encoder_output = rep
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
                if self.attr_bilstm_start is not None:
                    if self.compress_seq:
                        seqs_hiddens_packed_start, _ = self.attr_bilstm_start(seqs_rep_packed)
                        attr_seqs_hiddens_start, _ = pad_packed_sequence(seqs_hiddens_packed_start, batch_first=self.batch_first) # B, S, D
                        seqs_hiddens_packed_end, _ = self.attr_bilstm_end(seqs_rep_packed)
                        attr_seqs_hiddens_end, _ = pad_packed_sequence(seqs_hiddens_packed_end, batch_first=self.batch_first) # B, S, D
                    else:
                        attr_seqs_hiddens_start, _ = self.attr_bilstm_start(rep)
                        attr_seqs_hiddens_end, _ = self.attr_bilstm_end(rep)
                    attr_seqs_hiddens_start = torch.add(*attr_seqs_hiddens_start.chunk(2, dim=-1))
                    attr_seqs_hiddens_end = torch.add(*attr_seqs_hiddens_end.chunk(2, dim=-1))
            elif self.attr_bilstm_start is not None:
                if self.compress_seq:
                    att_mask = args[-1]
                    seqs_length = att_mask.sum(dim=-1).detach().cpu()
                    seqs_rep_packed = pack_padded_sequence(rep, seqs_length, batch_first=self.batch_first)
                    seqs_hiddens_packed_start, _ = self.attr_bilstm_start(seqs_rep_packed)
                    attr_seqs_hiddens_start, _ = pad_packed_sequence(seqs_hiddens_packed_start, batch_first=self.batch_first) # B, S, D
                    seqs_hiddens_packed_end, _ = self.attr_bilstm_end(seqs_rep_packed)
                    attr_seqs_hiddens_end, _ = pad_packed_sequence(seqs_hiddens_packed_end, batch_first=self.batch_first) # B, S, D
                else:
                    attr_seqs_hiddens_start, _ = self.attr_bilstm_start(rep)
                    attr_seqs_hiddens_end, _ = self.attr_bilstm_end(rep)
                attr_seqs_hiddens_start = torch.add(*attr_seqs_hiddens_start.chunk(2, dim=-1))
                attr_seqs_hiddens_end = torch.add(*attr_seqs_hiddens_end.chunk(2, dim=-1))
        
        return span_seqs_hiddens, attr_seqs_hiddens_start, attr_seqs_hiddens_end



class Base_BILSTM_Attr_Boundary(nn.Module):
    def __init__(self, sequence_encoder, tag2id, compress_seq=False, share_lstm=False, 
                    tagscheme='bmoes', batch_first=True, dropout_rate=0.3):
        """
        Args:
            sequence_encoder (nn.Module): encoder of sequence
            span2id (dict): map from span(et. B, I, O) to id
            tag2id (dict): map from attr(et. PER, LOC, ORG) to id
            compress_seq (bool, optional): whether compress sequence for lstm. Defaults to True.
            share_lstm (bool, optional): whether make span and attr share the same lstm after encoder. Defaults to False.
            span_use_lstm (bool, optional): whether add span lstm layer. Defaults to True.
            span_use_lstm (bool, optional): whether add attr lstm layer. Defaults to False.
            span_use_crf (bool, optional): whether add span crf layer. Defaults to True.
            batch_first (bool, optional): whether fisrt dim is batch. Defaults to True.
            dropout_rate (float, optional): dropout rate. Defaults to 0.3.
        """
        
        super(Base_BILSTM_Attr_Boundary, self).__init__()
        self.batch_first = batch_first
        self.tagscheme = tagscheme
        self.compress_seq = compress_seq
        self.sequence_encoder = sequence_encoder
        self.mlp_attr_start = nn.Linear(sequence_encoder.hidden_size, len(tag2id))
        self.mlp_attr_end = nn.Linear(sequence_encoder.hidden_size, len(tag2id))
        self.dropout = nn.Dropout(dropout_rate)
        self.tag2id = tag2id
        self.id2tag = {}
        for tag, tid in tag2id.items():
            self.id2tag[tid] = tag

        self.attr_start_bilstm = None
        self.attr_end_bilstm = None
        self.share_bilstm = None
        if share_lstm:
            self.share_bilstm = nn.LSTM(input_size=sequence_encoder.hidden_size, 
                                hidden_size=sequence_encoder.hidden_size, 
                                num_layers=1, 
                                bidirectional=True, 
                                batch_first=batch_first)
        else:
            self.attr_start_bilstm = nn.LSTM(input_size=sequence_encoder.hidden_size, 
                                        hidden_size=sequence_encoder.hidden_size, 
                                        num_layers=1, 
                                        bidirectional=True, 
                                        batch_first=batch_first)
            
            self.attr_end_bilstm = nn.LSTM(input_size=sequence_encoder.hidden_size, 
                                        hidden_size=sequence_encoder.hidden_size, 
                                        num_layers=1, 
                                        bidirectional=True, 
                                        batch_first=batch_first)


    def forward(self, *args):
        if not hasattr(self, '_flattened'):
            if self.share_bilstm is not None:
                self.share_bilstm.flatten_parameters()
            if self.attr_start_bilstm is not None:
                self.attr_start_bilstm.flatten_parameters()
            if self.attr_end_bilstm is not None:
                self.attr_end_bilstm.flatten_parameters()
            setattr(self, '_flattened', True)
        rep = self.sequence_encoder(*args) # B, S, D
        self.encoder_output = rep
        if self.share_bilstm is not None:
            if self.compress_seq:
                att_mask = args[-1]
                seqs_length = att_mask.sum(dim=-1).detach().cpu()
                seqs_rep_packed = pack_padded_sequence(rep, seqs_length, batch_first=self.batch_first)
                seqs_hiddens_packed, _ = self.share_bilstm(seqs_rep_packed)
                seqs_hiddens, _ = pad_packed_sequence(seqs_hiddens_packed, batch_first=self.batch_first) # B, S, D
            else:
                seqs_hiddens, _ = self.share_bilstm(rep)
            attr_start_hiddens = torch.add(*seqs_hiddens.chunk(2, dim=-1))
            attr_end_hiddens = attr_start_hiddens
        else:
            if self.compress_seq:
                att_mask = args[-1]
                seqs_length = att_mask.sum(dim=-1).detach().cpu()
                seqs_rep_packed = pack_padded_sequence(rep, seqs_length, batch_first=self.batch_first)
                seqs_hiddens_packed, _ = self.attr_start_bilstm(seqs_rep_packed)
                attr_start_hiddens, _ = pad_packed_sequence(seqs_hiddens_packed, batch_first=self.batch_first) # B, S ,D
                seqs_hiddens_packed, _ = self.attr_end_bilstm(seqs_rep_packed)
                attr_end_hiddens, _ = pad_packed_sequence(seqs_hiddens_packed, batch_first=self.batch_first) # B, S ,D
            else:
                attr_start_hiddens, _ = self.attr_start_bilstm(rep)
                attr_end_hiddens, _ = self.attr_end_bilstm(rep)
            attr_start_hiddens = torch.add(*attr_start_hiddens.chunk(2, dim=-1))
            attr_end_hiddens = torch.add(*attr_end_hiddens.chunk(2, dim=-1))
        
        return attr_start_hiddens, attr_end_hiddens
