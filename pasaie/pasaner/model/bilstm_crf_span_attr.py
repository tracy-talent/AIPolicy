"""
 Author: liujian
 Date: 2020-10-26 17:48:40
 Last Modified by: liujian
 Last Modified time: 2020-10-26 17:48:40
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..encoder import BERTEncoder
from ..decoder import CRF
from ...utils.entity_extract import *
from ...module.nn.attention import AdditiveAttention, DotProductAttention, MultiplicativeAttention
from ...module.nn import PoolerStartLogits, PoolerEndLogits


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
        seq_len = items[-1].sum().item()
        if self.crf_span is not None:
            preds_span = self.crf_span.decode(logits_span, mask=items[-1])[0]
        else:
            preds_span = probs_span[:,:seq_len,:].argmax(dim=-1).squeeze().detach().cpu().numpy()
        if self.crf_attr is not None:
            preds_attr = self.crf_attr.decode(logits_attr, mask=items[-1])[0]
        else:
            preds_attr = probs_attr[:,:seq_len,:].argmax(dim=-1).squeeze().deatch().cpu().numpy()
        
        spos, tpos = 0, seq_len
        if 'bert' in self.sequence_encoder.__class__.__name__.lower():
            spos, tpos = 1, -1
        spans = [self.id2span[sid] for sid in preds_span[spos:tpos]] # 包含'[CLS]'和'[SEP]'
        attrs = [self.id2attr[aid] for aid in preds_attr[spos:tpos]]
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


class BILSTM_CRF_Span_Attr_Boundary(nn.Module):
    def __init__(self, sequence_encoder, span2id, attr2id, compress_seq=False, share_lstm=False, span_use_lstm=True, attr_use_lstm=False, span_use_crf=True, tagscheme='bmoes', batch_first=True):
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
        """
        
        super(BILSTM_CRF_Span_Attr_Boundary, self).__init__()
        self.batch_first = batch_first
        self.compress_seq = compress_seq
        self.tagscheme = tagscheme
        self.sequence_encoder = sequence_encoder
        self.mlp_span = nn.Linear(sequence_encoder.hidden_size, len(span2id))
        self.mlp_attr_start = nn.Linear(sequence_encoder.hidden_size, len(attr2id))
        self.mlp_attr_end = nn.Linear(sequence_encoder.hidden_size, len(attr2id))
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

        logits_span = self.mlp_span(span_seqs_hiddens) # B, S, V
        logits_attr_start = self.mlp_attr_start(attr_seqs_hiddens) # B, S, V
        logits_attr_end = self.mlp_attr_end(attr_seqs_hiddens) # B, S, V
        
        return logits_span, logits_attr_start, logits_attr_end


class BILSTM_CRF_Span_Attr_Boundary_StartPrior(nn.Module):
    def __init__(self, sequence_encoder, span2id, attr2id, compress_seq=False, share_lstm=False, span_use_lstm=True, attr_use_lstm=False, span_use_crf=True, tagscheme='bmoes', batch_first=True, soft_label=False, dropout_rate=0.1):
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
            soft_label (bool, optional): use one hot if soft_label is True. Defaults to False.
            dropout_rate (float, optional): dropout rate. Defaults to 0.1.
        """
        
        super(BILSTM_CRF_Span_Attr_Boundary_StartPrior, self).__init__()
        self.batch_first = batch_first
        self.compress_seq = compress_seq
        self.tagscheme = tagscheme
        self.soft_label = soft_label
        self.sequence_encoder = sequence_encoder
        self.num_labels = len(attr2id)
        self.mlp_span = nn.Linear(sequence_encoder.hidden_size, len(span2id))
        self.mlp_attr_start = PoolerStartLogits(sequence_encoder.hidden_size, self.num_labels)
        if self.soft_label:
            self.mlp_attr_end = PoolerEndLogits(sequence_encoder.hidden_size + self.num_labels, self.num_labels)
        else:
            self.mlp_attr_end = PoolerEndLogits(sequence_encoder.hidden_size + 1, self.num_labels)
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


    def forward(self, labels_attr_start=None, *args):
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

        span_seqs_hiddens = self.dropout(span_seqs_hiddens)
        attr_seqs_hiddens = self.dropout(attr_seqs_hiddens)
        logits_span = self.mlp_span(span_seqs_hiddens) # B, S, V
        logits_attr_start = self.mlp_attr_start(attr_seqs_hiddens) # B, S, V
        if labels_attr_start is not None and self.training:
            if self.soft_label:
                attr_start_labels_logits = torch.zeros_like(logits_attr_start).to(labels_attr_start.device)
                attr_start_labels_logits.scatter_(2, labels_attr_start.unsqueeze(2), 1)
            else:
                attr_start_labels_logits = labels_attr_start.unsqueeze(2).float()
        else:
            if self.soft_label:
                preds_attr_start = logits_attr_start.argmax(dim=-1, keepdims=True)
                attr_start_labels_logits = torch.zeros_like(logits_attr_start).to(preds_attr_start.device)
                attr_start_labels_logits.scatter_(2, preds_attr_start, 1)
            else:
                attr_start_labels_logits = logits_attr_start.argmax(dim=-1, keepdims=True).float()
        logits_attr_end = self.mlp_attr_end(attr_seqs_hiddens, attr_start_labels_logits)
        
        return logits_span, logits_attr_start, logits_attr_end



class BILSTM_CRF_Span_Attr_Boundary_Attention(nn.Module):
    def __init__(self, sequence_encoder, span2id, attr2id, compress_seq=False, share_lstm=False, span_use_lstm=True, attr_use_lstm=False, span_use_crf=True, tagscheme='bmoes', batch_first=True):
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
        """
        
        super(BILSTM_CRF_Span_Attr_Boundary_Attention, self).__init__()
        self.batch_first = batch_first
        self.compress_seq = compress_seq
        self.tagscheme = tagscheme
        self.sequence_encoder = sequence_encoder
        self.attention = AdditiveAttention(sequence_encoder.hidden_size, sequence_encoder.hidden_size, sequence_encoder.hidden_size)
        self.mlp_span2attr = nn.Linear(sequence_encoder.hidden_size, sequence_encoder.hidden_size)
        self.mlp_span = nn.Linear(sequence_encoder.hidden_size, len(span2id))
        self.mlp_attr_start = nn.Linear(sequence_encoder.hidden_size * 2, len(attr2id))
        self.mlp_attr_end = nn.Linear(sequence_encoder.hidden_size * 2, len(attr2id))
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

    def dot_product_attention(self, attention_kv, attention_query):
        """dot product attention for attr_hidden_state to attend span_hidden_state.

        Args:
            attention_kv (torch.Tensor): key and value matrix of attention mechanism, size(B, S, H).
            attention_query (torch.Tensor): query matrix of attention mechanism, size(B, S, H)

        Returns:
            attention_output (torch.Tensor): attention output matrix, size(B, S, H).
            attention_weight (torch.Tensor): attention weight matrix, size(B, S, S).
        """
        attention_score = torch.matmul(attention_query, attention_kv.transpose(1, 2))
        attention_weight = F.softmax(attention_score, dim=-1)
        attention_output = torch.matmul(attention_weight, attention_kv)
        return attention_output, attention_weight

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

        # dot product attention
        # span_attention_output = self.dot_product_attention(span_seqs_hiddens, attr_seqs_hiddens)[0] # accelerate
        # additive attention
        # attr_start_attention_output = [self.attention(span_seqs_hiddens, attr_seqs_hiddens[:,i,:])[0] 
        #                                     for i in range(attr_seqs_hiddens.size(1))]
        # attr_start_attention_output = torch.stack(attr_start_attention_output, dim=1)
        span_attention_output = self.attention(span_seqs_hiddens, attr_seqs_hiddens)[0] # accelerate
        attr_attention_output = torch.tanh(self.mlp_span2attr(span_attention_output))
        attr_seqs_hiddens = torch.cat([attr_seqs_hiddens, attr_attention_output], dim=-1)
        logits_span = self.mlp_span(span_seqs_hiddens) # B, S, V
        logits_attr_start = self.mlp_attr_start(attr_seqs_hiddens) # B, S, V
        logits_attr_end = self.mlp_attr_end(attr_seqs_hiddens) # B, S, V
        
        return logits_span, logits_attr_start, logits_attr_end
