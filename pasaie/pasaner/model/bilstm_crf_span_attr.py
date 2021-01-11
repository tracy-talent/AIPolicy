"""
 Author: liujian
 Date: 2020-10-26 17:48:40
 Last Modified by: liujian
 Last Modified time: 2020-10-26 17:48:40
"""

from .base_model import Base_BILSTM_CRF_Span_Attr, Base_BILSTM_CRF_Span_Attr_Three
from ..decoder import CRF
from ...utils.entity_extract import *
from ...module.nn.attention import AdditiveAttention, DotProductAttention, MultiplicativeAttention
from ...module.nn.linear import PoolerStartLogits, PoolerEndLogits, Linear3D, LinearSequence

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class BILSTM_CRF_Span_Attr_Tail(nn.Module):
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
        super(BILSTM_CRF_Span_Attr_Tail, self).__init__(
            sequence_encoder=sequence_encoder,
            span2id=span2id,
            attr2id=attr2id,
            compress_seq=compress_seq,
            share_lstm=share_lstm,
            span_use_lstm=span_use_lstm,
            attr_use_lstm=attr_use_lstm,
            span_use_crf=span_use_crf,
            tagscheme=tagscheme,
            batch_first=batch_first,
            dropout_rate=dropout_rate
        )
        self.mlp_attr = nn.Linear(sequence_encoder.hidden_size, len(attr2id))
        del self.mlp_attr_start
        del self.mlp_attr_end
    

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
        span_seqs_hiddens, attr_seqs_hiddens = super(BILSTM_CRF_Span_Attr_Tail, self).forward(*args)
        # dropout layer
        span_seqs_hiddens = self.dropout(span_seqs_hiddens)
        attr_seqs_hiddens = self.dropout(attr_seqs_hiddebs)
        # reduce entity information to end token
        if span_labels is not None:
            span_bid, span_eid = self.span2id['B'], self.span2id['E']
            spos = -1
            for i in range(span_labels.size(0)):
                for j in range(span_labels.size(1)):
                    if span_labels[i][j].item() == span_bid:
                        spos = j
                    elif span_labels[i][j].item() == span_eid:
                        attr_seqs_hiddens[i][j] = torch.mean(attr_seqs_hiddens[i][spos:j+1], dim=0)
        # output layer
        logits_span = self.mlp_span(span_seqs_hiddens) # B, S, V
        logits_attr = self.mlp_attr(attr_seqs_hiddens) # B, S, V
        
        return logits_span, logits_attr



class BILSTM_CRF_Span_Attr_Boundary(Base_BILSTM_CRF_Span_Attr):
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
        super(BILSTM_CRF_Span_Attr_Boundary, self).__init__(
            sequence_encoder=sequence_encoder,
            span2id=span2id,
            attr2id=attr2id,
            compress_seq=compress_seq,
            share_lstm=share_lstm,
            span_use_lstm=span_use_lstm,
            attr_use_lstm=attr_use_lstm,
            span_use_crf=span_use_crf,
            tagscheme=tagscheme,
            batch_first=batch_first,
            dropout_rate=dropout_rate
        )


    def forward(self, *args):
        span_seqs_hiddens, attr_seqs_hiddens = super(BILSTM_CRF_Span_Attr_Boundary, self).forward(*args)
        # dropout layer
        span_seqs_hiddens = self.dropout(span_seqs_hiddens)
        attr_seqs_hiddens = self.dropout(attr_seqs_hiddens)
        # output layer
        logits_span = self.mlp_span(span_seqs_hiddens) # B, S, V
        logits_attr_start = self.mlp_attr_start(attr_seqs_hiddens) # B, S, V
        logits_attr_end = self.mlp_attr_end(attr_seqs_hiddens) # B, S, V
        
        return logits_span, logits_attr_start, logits_attr_end



class BILSTM_CRF_Span_Attr_Boundary_StartPrior(Base_BILSTM_CRF_Span_Attr):
    def __init__(self, sequence_encoder, span2id, attr2id, compress_seq=False, share_lstm=False, span_use_lstm=True, attr_use_lstm=False, span_use_crf=True, tagscheme='bmoes', batch_first=True, soft_label=False, dropout_rate=0.3):
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
            dropout_rate (float, optional): dropout rate. Defaults to 0.3.
        """
        super(BILSTM_CRF_Span_Attr_Boundary_StartPrior, self).__init__(
            sequence_encoder=sequence_encoder,
            span2id=span2id,
            attr2id=attr2id,
            compress_seq=compress_seq,
            share_lstm=share_lstm,
            span_use_lstm=span_use_lstm,
            attr_use_lstm=attr_use_lstm,
            span_use_crf=span_use_crf,
            tagscheme=tagscheme,
            batch_first=batch_first,
            dropout_rate=dropout_rate
        )
        self.soft_label = soft_label
        num_labels = len(attr2id)
        self.mlp_attr_start = PoolerStartLogits(sequence_encoder.hidden_size, num_labels)
        if self.soft_label:
            self.mlp_attr_end = PoolerEndLogits(sequence_encoder.hidden_size + num_labels, num_labels)
        else:
            self.mlp_attr_end = PoolerEndLogits(sequence_encoder.hidden_size + 1, num_labels)
    

    def forward(self, labels_attr_start=None, *args):
        span_seqs_hiddens, attr_seqs_hiddens = super(BILSTM_CRF_Span_Attr_Boundary_StartPrior, self).forward(*args)
        # dropout layer
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



class BILSTM_CRF_Span_Attr_Boundary_Attention(Base_BILSTM_CRF_Span_Attr):
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
        
        super(BILSTM_CRF_Span_Attr_Boundary_Attention, self).__init__(
            sequence_encoder=sequence_encoder,
            span2id=span2id,
            attr2id=attr2id,
            compress_seq=compress_seq,
            share_lstm=share_lstm,
            span_use_lstm=span_use_lstm,
            attr_use_lstm=attr_use_lstm,
            span_use_crf=span_use_crf,
            tagscheme=tagscheme,
            batch_first=batch_first,
            dropout_rate=dropout_rate
        )
        # self.attention = AdditiveAttention(sequence_encoder.hidden_size, sequence_encoder.hidden_size, sequence_encoder.hidden_size)
        self.mlp_span2attr = nn.Linear(sequence_encoder.hidden_size, sequence_encoder.hidden_size)
        self.mlp_attr_start = nn.Linear(sequence_encoder.hidden_size * 2, len(attr2id))
        self.mlp_attr_end = nn.Linear(sequence_encoder.hidden_size * 2, len(attr2id))

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
        span_seqs_hiddens, attr_seqs_hiddens = super(BILSTM_CRF_Span_Attr_Boundary_Attention, self).forward(*args)
        # dot product attention
        span_attention_output = self.dot_product_attention(span_seqs_hiddens, attr_seqs_hiddens)[0]
        # additive attention
        # span_attention_output = self.attention(span_seqs_hiddens, attr_seqs_hiddens)[0]
        attr_attention_output = torch.tanh(self.mlp_span2attr(span_attention_output))
        attr_seqs_hiddens = torch.cat([attr_seqs_hiddens, attr_attention_output], dim=-1)
        # dropout layer
        span_seqs_hiddens = self.dropout(span_seqs_hiddens)
        attr_seqs_hiddens = self.dropout(attr_seqs_hiddens)
        # output layer
        logits_span = self.mlp_span(span_seqs_hiddens) # B, S, V
        logits_attr_start = self.mlp_attr_start(attr_seqs_hiddens) # B, S, V
        logits_attr_end = self.mlp_attr_end(attr_seqs_hiddens) # B, S, V
        
        return logits_span, logits_attr_start, logits_attr_end



class BILSTM_CRF_Span_Attr_Three_Boundary_Attention(Base_BILSTM_CRF_Span_Attr_Three):
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
        
        super(BILSTM_CRF_Span_Attr_Three_Boundary_Attention, self).__init__(
            sequence_encoder=sequence_encoder,
            span2id=span2id,
            attr2id=attr2id,
            compress_seq=compress_seq,
            share_lstm=share_lstm,
            span_use_lstm=span_use_lstm,
            attr_use_lstm=attr_use_lstm,
            span_use_crf=span_use_crf,
            tagscheme=tagscheme,
            batch_first=batch_first,
            dropout_rate=dropout_rate
        )
        # self.attention = AdditiveAttention(sequence_encoder.hidden_size, sequence_encoder.hidden_size, sequence_encoder.hidden_size)
        self.mlp_span2attr_start = nn.Linear(sequence_encoder.hidden_size, sequence_encoder.hidden_size)
        self.mlp_span2attr_end = nn.Linear(sequence_encoder.hidden_size, sequence_encoder.hidden_size)
        self.mlp_attr_start = nn.Linear(sequence_encoder.hidden_size * 2, len(attr2id))
        self.mlp_attr_end = nn.Linear(sequence_encoder.hidden_size * 2, len(attr2id))

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
        span_seqs_hiddens, attr_seqs_hiddens_start, attr_seqs_hiddens_end = super(BILSTM_CRF_Span_Attr_Three_Boundary_Attention, self).forward(*args)
        # dot product attention
        span_attention_output_start = self.dot_product_attention(span_seqs_hiddens, attr_seqs_hiddens_start)[0]
        span_attention_output_end = self.dot_product_attention(span_seqs_hiddens, attr_seqs_hiddens_end)[0]
        # additive attention
        # span_attention_output_start = self.attention(span_seqs_hiddens, attr_seqs_hiddens_start)[0]
        # span_attention_output_end = self.attention(span_seqs_hiddens, attr_seqs_hiddens_end)[0]
        attr_attention_output_start = torch.tanh(self.mlp_span2attr_start(span_attention_output_start))
        attr_attention_output_end = torch.tanh(self.mlp_span2attr_end(span_attention_output_end))
        attr_seqs_hiddens_start = torch.cat([attr_seqs_hiddens_start, attr_attention_output_start], dim=-1)
        attr_seqs_hiddens_end = torch.cat([attr_seqs_hiddens_end, attr_attention_output_end], dim=-1)
        # dropout layer
        span_seqs_hiddens = self.dropout(span_seqs_hiddens)
        attr_seqs_hiddens_start = self.dropout(attr_seqs_hiddens_start)
        attr_seqs_hiddens_end = self.dropout(attr_seqs_hiddens_end)
        # output layer
        logits_span = self.mlp_span(span_seqs_hiddens) # B, S, V
        logits_attr_start = self.mlp_attr_start(attr_seqs_hiddens_start) # B, S, V
        logits_attr_end = self.mlp_attr_end(attr_seqs_hiddens_end) # B, S, V
        
        return logits_span, logits_attr_start, logits_attr_end



class BILSTM_CRF_Span_Attr_Boundary_MMoE(Base_BILSTM_CRF_Span_Attr):
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
        
        super(BILSTM_CRF_Span_Attr_Boundary_MMoE, self).__init__(
            sequence_encoder=sequence_encoder,
            span2id=span2id,
            attr2id=attr2id,
            compress_seq=compress_seq,
            share_lstm=share_lstm,
            span_use_lstm=span_use_lstm,
            attr_use_lstm=attr_use_lstm,
            span_use_crf=span_use_crf,
            tagscheme=tagscheme,
            batch_first=batch_first,
            dropout_rate=dropout_rate
        )
        # self.moe_gate1 = LinearSequence(sequence_encoder.max_length, sequence_encoder.hidden_size, 2)
        # self.moe_gate2 = LinearSequence(sequence_encoder.max_length, sequence_encoder.hidden_size, 2)
        self.moe_gate1 = nn.Linear(sequence_encoder.hidden_size, 2)
        self.moe_gate2 = nn.Linear(sequence_encoder.hidden_size, 2)


    def expert_fusion(self, input_hiddens, *expert_hiddens):
        """dot product attention for attr_hidden_state to attend span_hidden_state.

        Args:
            attention_kv (torch.Tensor): key and value matrix of attention mechanism, size(B, S, H).
            attention_query (torch.Tensor): query matrix of attention mechanism, size(B, S, H)

        Returns:
            attention_output (torch.Tensor): attention output matrix, size(B, S, H).
            attention_weight (torch.Tensor): attention weight matrix, size(B, S, S).
        """
        gate1_expert_score = self.moe_gate1(input_hiddens)
        gate1_expert_weight = F.softmax(gate1_expert_score, dim=-1)
        gate2_expert_score = self.moe_gate2(input_hiddens)
        gate2_expert_weight = F.softmax(gate2_expert_score, dim=-1)
        expert_hiddens = torch.stack(expert_hiddens, dim=2)
        expert_fusion_output1 = torch.matmul(gate1_expert_weight.unsqueeze(2), expert_hiddens).squeeze(2)
        expert_fusion_output2 = torch.matmul(gate2_expert_weight.unsqueeze(2), expert_hiddens).squeeze(2)
        
        return expert_fusion_output1, expert_fusion_output2


    def forward(self, *args):
        span_seqs_hiddens, attr_seqs_hiddens = super(BILSTM_CRF_Span_Attr_Boundary_MMoE, self).forward(*args)
        span_seqs_hiddens, attr_seqs_hiddens = self.expert_fusion(self.encoder_output, span_seqs_hiddens, attr_seqs_hiddens)
        # dropout layer
        span_seqs_hiddens = self.dropout(span_seqs_hiddens)
        attr_seqs_hiddens = self.dropout(attr_seqs_hiddens)
        # output layer
        logits_span = self.mlp_span(span_seqs_hiddens) # B, S, V
        logits_attr_start = self.mlp_attr_start(attr_seqs_hiddens) # B, S, V
        logits_attr_end = self.mlp_attr_end(attr_seqs_hiddens) # B, S, V
        
        return logits_span, logits_attr_start, logits_attr_end




class BILSTM_CRF_Span_Attr_Three_Boundary_MMoE(Base_BILSTM_CRF_Span_Attr_Three):
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
        
        super(BILSTM_CRF_Span_Attr_Three_Boundary_MMoE, self).__init__(
            sequence_encoder=sequence_encoder,
            span2id=span2id,
            attr2id=attr2id,
            compress_seq=compress_seq,
            share_lstm=share_lstm,
            span_use_lstm=span_use_lstm,
            attr_use_lstm=attr_use_lstm,
            span_use_crf=span_use_crf,
            tagscheme=tagscheme,
            batch_first=batch_first,
            dropout_rate=dropout_rate
        )
        # self.moe_gate1 = LinearSequence(sequence_encoder.max_length, sequence_encoder.hidden_size, 2)
        # self.moe_gate2 = LinearSequence(sequence_encoder.max_length, sequence_encoder.hidden_size, 2)
        # self.moe_gate3 = LinearSequence(sequence_encoder.max_length, sequence_encoder.hidden_size, 2)
        self.moe_gate1 = nn.Linear(sequence_encoder.hidden_size, 2)
        self.moe_gate2 = nn.Linear(sequence_encoder.hidden_size, 2)
        self.moe_gate3 = nn.Linear(sequence_encoder.hidden_size, 2)


    def expert_fusion(self, input_hiddens, *expert_hiddens):
        """dot product attention for attr_hidden_state to attend span_hidden_state.

        Args:
            attention_kv (torch.Tensor): key and value matrix of attention mechanism, size(B, S, H).
            attention_query (torch.Tensor): query matrix of attention mechanism, size(B, S, H)

        Returns:
            attention_output (torch.Tensor): attention output matrix, size(B, S, H).
            attention_weight (torch.Tensor): attention weight matrix, size(B, S, S).
        """
        gate1_expert_score = self.moe_gate1(input_hiddens)
        gate1_expert_weight = F.softmax(gate1_expert_score, dim=-1)
        gate2_expert_score = self.moe_gate2(input_hiddens)
        gate2_expert_weight = F.softmax(gate2_expert_score, dim=-1)
        gate3_expert_score = self.moe_gate3(input_hiddens)
        gate3_expert_weight = F.softmax(gate3_expert_score, dim=-1)
        expert_hiddens = torch.stack(expert_hiddens, dim=2)
        expert_fusion_output1 = torch.matmul(gate1_expert_weight.unsqueeze(2), expert_hiddens).squeeze(2)
        expert_fusion_output2 = torch.matmul(gate2_expert_weight.unsqueeze(2), expert_hiddens).squeeze(2)
        expert_fusion_output3 = torch.matmul(gate3_expert_weight.unsqueeze(2), expert_hiddens).squeeze(2)
        
        return expert_fusion_output1, expert_fusion_output2, expert_fusion_output3


    def forward(self, *args):
        span_seqs_hiddens, attr_seqs_hiddens_start, attr_seqs_hiddens_end = super(BILSTM_CRF_Span_Attr_Three_Boundary_MMoE, self).forward(*args)
        span_seqs_hiddens, attr_seqs_hiddens_start, attr_seqs_hiddens_end = self.expert_fusion(self.encoder_output, span_seqs_hiddens, attr_seqs_hiddens_start, attr_seqs_hiddens_end)
        # dropout layer
        span_seqs_hiddens = self.dropout(span_seqs_hiddens)
        attr_seqs_hiddens_start = self.dropout(attr_seqs_hiddens_start)
        attr_seqs_hiddens_end = self.dropout(attr_seqs_hiddens_end)
        # output layer
        logits_span = self.mlp_span(span_seqs_hiddens) # B, S, V
        logits_attr_start = self.mlp_attr_start(attr_seqs_hiddens_start) # B, S, V
        logits_attr_end = self.mlp_attr_end(attr_seqs_hiddens_end) # B, S, V
        
        return logits_span, logits_attr_start, logits_attr_end


        
class BILSTM_CRF_Span_Attr_Boundary_PLE(Base_BILSTM_CRF_Span_Attr):
    def __init__(self, sequence_encoder, span2id, attr2id, compress_seq=False, share_lstm=False, span_use_lstm=True, attr_use_lstm=False, span_use_crf=True, tagscheme='bmoes', batch_first=True, dropout_rate=0.3, experts_layers=2, experts_num=2):
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
        
        super(BILSTM_CRF_Span_Attr_Boundary_PLE, self).__init__(
            sequence_encoder=sequence_encoder,
            span2id=span2id,
            attr2id=attr2id,
            compress_seq=compress_seq,
            share_lstm=share_lstm,
            span_use_lstm=span_use_lstm,
            attr_use_lstm=attr_use_lstm,
            span_use_crf=span_use_crf,
            tagscheme=tagscheme,
            batch_first=batch_first,
            dropout_rate=dropout_rate
        )
        self.experts_layers = experts_layers
        self.experts_num = experts_num
        self.selector_num = 2
        hidden_size = sequence_encoder.hidden_size
        self.layers_experts_shared = nn.ModuleList()
        self.layers_experts_task1 = nn.ModuleList()
        self.layers_experts_task2 = nn.ModuleList()
        self.layers_experts_shared_gate = nn.ModuleList()
        self.layers_experts_task1_gate = nn.ModuleList()
        self.layers_experts_task2_gate = nn.ModuleList()
        self.layers_layernorm = nn.ModuleList([nn.ModuleList() for _ in range(3)])
        for i in range(experts_layers):
            # experts shared
            self.layers_experts_shared.append(Linear3D(hidden_size, hidden_size, experts_num))        

            # experts task1
            self.layers_experts_task1.append(Linear3D(hidden_size, hidden_size, experts_num))            

            # experts task2
            self.layers_experts_task2.append(Linear3D(hidden_size, hidden_size, experts_num))       

            # gates shared
            self.layers_experts_shared_gate.append(nn.Linear(hidden_size, experts_num * 3))            

            # gate task1
            self.layers_experts_task1_gate.append(nn.Linear(hidden_size, experts_num * self.selector_num))            

            # gate task2
            self.layers_experts_task2_gate.append(nn.Linear(hidden_size, experts_num * self.selector_num))            

            # layer norm
            for j in range(3):
                self.layers_layernorm[i].append(nn.LayerNorm(hidden_size, elementwise_affine=False))


    def progressive_layered_extraction(self, gate_shared_output_final, gate_task1_output_final, gate_task2_output_final):
        for i in range(self.experts_layers):
            # shared  output
            experts_shared_output = torch.relu(self.layers_experts_shared[i](gate_shared_output_final))
            # experts_shared_output = self.layers_experts_shared[i](gate_shared_output_final)
            experts_shared_output = F.dropout(experts_shared_output, p=0.1)

            # task1 output
            experts_task1_output = torch.relu(self.layers_experts_task1[i](gate_task1_output_final))
            # experts_task1_output = self.layers_experts_task1[i](gate_task1_output_final)
            experts_task1_output = F.dropout(experts_task1_output, p=0.1)

            # task2 output
            experts_task2_output = torch.relu(self.layers_experts_task2[i](gate_task2_output_final))
            # experts_task2_output = self.layers_experts_task2[i](gate_task2_output_final)
            experts_task2_output = F.dropout(experts_task2_output, p=0.1)

            # gate shared output
            gate_shared_output = self.layers_experts_shared_gate[i](gate_shared_output_final) # (B, S, C)
            gate_shared_output = F.softmax(gate_shared_output, dim=-1)
            gate_shared_output = torch.matmul(gate_shared_output.unsqueeze(-2), 
                                    torch.cat([experts_task1_output, experts_shared_output, experts_task2_output], dim=-1).permute(0, 1, 3, 2)).squeeze(-2)
            # gate_shared_output = F.dropout(gate_shared_output, p=0.1)
            gate_shared_output_final = gate_shared_output
            # gate_shared_output_final = self.layers_layernorm[i][0](gate_shared_output + gate_shared_output_final)

            # gate task1 output
            gate_task1_output = self.layers_experts_task1_gate[i](gate_task1_output_final) # (B, S, C)
            gate_task1_output = F.softmax(gate_task1_output, dim=-1)
            gate_task1_output = torch.matmul(gate_task1_output.unsqueeze(-2), 
                                    torch.cat([experts_task1_output, experts_shared_output], dim=-1).permute(0, 1, 3, 2)).squeeze(-2)
            # gate_task1_output = F.dropout(gate_task1_output, p=0.1)
            gate_task1_output_final = gate_task1_output
            # gate_task1_output_final = self.layers_layernorm[i][1](gate_task1_output + gate_task1_output_final)

            # gate task2 output
            gate_task2_output = self.layers_experts_task2_gate[i](gate_task2_output_final) # (B, S, C)
            gate_task2_output = F.softmax(gate_task2_output, dim=-1)
            gate_task2_output = torch.matmul(gate_task2_output.unsqueeze(-2), 
                                    torch.cat([experts_task2_output, experts_shared_output], dim=-1).permute(0, 1, 3, 2)).squeeze(-2)
            # gate_task2_output = F.dropout(gate_task2_output, p=0.1)
            gate_task2_output_final = gate_task2_output
            # gate_task2_output_final = self.layers_layernorm[i][2](gate_task2_output + gate_task2_output_final)
        
        return gate_shared_output_final, gate_task1_output_final, gate_task2_output_final


    def forward(self, *args):
        span_seqs_hiddens, attr_seqs_hiddens = super(BILSTM_CRF_Span_Attr_Boundary_PLE, self).forward(*args)
        _, span_seqs_hiddens, attr_seqs_hiddens = self.progressive_layered_extraction(self.encoder_output, span_seqs_hiddens, attr_seqs_hiddens)
        # dropout layer
        span_seqs_hiddens = self.dropout(span_seqs_hiddens)
        attr_seqs_hiddens = self.dropout(attr_seqs_hiddens)
        # output layer
        logits_span = self.mlp_span(span_seqs_hiddens) # B, S, V
        logits_attr_start = self.mlp_attr_start(attr_seqs_hiddens) # B, S, V
        logits_attr_end = self.mlp_attr_end(attr_seqs_hiddens) # B, S, V
        
        return logits_span, logits_attr_start, logits_attr_end


class BILSTM_CRF_Span_Attr_Boundary_PLE_1(BILSTM_CRF_Span_Attr_Boundary_PLE):
    def __init__(self, sequence_encoder, span2id, attr2id, compress_seq=False, share_lstm=False, span_use_lstm=True, attr_use_lstm=False, span_use_crf=True, tagscheme='bmoes', batch_first=True, dropout_rate=0.3, experts_layers=2, experts_num=2):
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
        
        super(BILSTM_CRF_Span_Attr_Boundary_PLE_1, self).__init__(
            sequence_encoder=sequence_encoder,
            span2id=span2id,
            attr2id=attr2id,
            compress_seq=compress_seq,
            share_lstm=share_lstm,
            span_use_lstm=span_use_lstm,
            attr_use_lstm=attr_use_lstm,
            span_use_crf=span_use_crf,
            tagscheme=tagscheme,
            batch_first=batch_first,
            dropout_rate=dropout_rate,
            experts_layers=experts_layers,
            experts_num=experts_num
        )


    def progressive_layered_extraction(self, gate_shared_output_final, gate_task1_output_final, gate_task2_output_final):
        for i in range(self.experts_layers):
            # shared  output
            # experts_shared_output = torch.relu(self.layers_experts_shared[i](gate_shared_output_final))
            experts_shared_output = self.layers_experts_shared[i](gate_shared_output_final)
            # experts_shared_output = F.dropout(experts_shared_output, p=0.1)

            # task1 output
            # experts_task1_output = torch.relu(self.layers_experts_task1[i](gate_task1_output_final))
            experts_task1_output = self.layers_experts_task1[i](gate_task1_output_final)
            # experts_task1_output = F.dropout(experts_task1_output, p=0.1)

            # task2 output
            # experts_task2_output = torch.relu(self.layers_experts_task2[i](gate_task2_output_final))
            experts_task2_output = self.layers_experts_task2[i](gate_task2_output_final)
            # experts_task2_output = F.dropout(experts_task2_output, p=0.1)

            # gate shared output
            gate_shared_output = self.layers_experts_shared_gate[i](gate_shared_output_final) # (B, S, C)
            gate_shared_output = F.softmax(gate_shared_output, dim=-1)
            gate_shared_output = torch.matmul(gate_shared_output.unsqueeze(-2), 
                                    torch.cat([experts_task1_output, experts_shared_output, experts_task2_output], dim=-1).permute(0, 1, 3, 2)).squeeze(-2)
            # gate_shared_output = F.dropout(gate_shared_output, p=0.1)
            gate_shared_output_final = gate_shared_output
            # gate_shared_output_final = self.layers_layernorm[i][0](gate_shared_output + gate_shared_output_final)

            # gate task1 output
            gate_task1_output = self.layers_experts_task1_gate[i](gate_task1_output_final) # (B, S, C)
            gate_task1_output = F.softmax(gate_task1_output, dim=-1)
            gate_task1_output = torch.matmul(gate_task1_output.unsqueeze(-2), 
                                    torch.cat([experts_task1_output, experts_shared_output], dim=-1).permute(0, 1, 3, 2)).squeeze(-2)
            # gate_task1_output = F.dropout(gate_task1_output, p=0.1)
            gate_task1_output_final = gate_task1_output
            # gate_task1_output_final = self.layers_layernorm[i][1](gate_task1_output + gate_task1_output_final)

            # gate task2 output
            gate_task2_output = self.layers_experts_task2_gate[i](gate_task2_output_final) # (B, S, C)
            gate_task2_output = F.softmax(gate_task2_output, dim=-1)
            gate_task2_output = torch.matmul(gate_task2_output.unsqueeze(-2), 
                                    torch.cat([experts_task2_output, experts_shared_output], dim=-1).permute(0, 1, 3, 2)).squeeze(-2)
            # gate_task2_output = F.dropout(gate_task2_output, p=0.1)
            gate_task2_output_final = gate_task2_output
            # gate_task2_output_final = self.layers_layernorm[i][2](gate_task2_output + gate_task2_output_final)
        
        return gate_shared_output_final, gate_task1_output_final, gate_task2_output_final



class BILSTM_CRF_Span_Attr_Boundary_PLE_2(BILSTM_CRF_Span_Attr_Boundary_PLE):
    def __init__(self, sequence_encoder, span2id, attr2id, compress_seq=False, share_lstm=False, span_use_lstm=True, attr_use_lstm=False, span_use_crf=True, tagscheme='bmoes', batch_first=True, dropout_rate=0.3, experts_layers=2, experts_num=2):
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
        
        super(BILSTM_CRF_Span_Attr_Boundary_PLE_2, self).__init__(
            sequence_encoder=sequence_encoder,
            span2id=span2id,
            attr2id=attr2id,
            compress_seq=compress_seq,
            share_lstm=share_lstm,
            span_use_lstm=span_use_lstm,
            attr_use_lstm=attr_use_lstm,
            span_use_crf=span_use_crf,
            tagscheme=tagscheme,
            batch_first=batch_first,
            dropout_rate=dropout_rate,
            experts_layers=experts_layers,
            experts_num=experts_num
        )


    def progressive_layered_extraction(self, gate_shared_output_final, gate_task1_output_final, gate_task2_output_final):
        for i in range(self.experts_layers):
            # shared  output
            experts_shared_output = torch.relu(self.layers_experts_shared[i](gate_shared_output_final))
            # experts_shared_output = self.layers_experts_shared[i](gate_shared_output_final)
            # experts_shared_output = F.dropout(experts_shared_output, p=0.1)

            # task1 output
            experts_task1_output = torch.relu(self.layers_experts_task1[i](gate_task1_output_final))
            # experts_task1_output = self.layers_experts_task1[i](gate_task1_output_final)
            # experts_task1_output = F.dropout(experts_task1_output, p=0.1)

            # task2 output
            experts_task2_output = torch.relu(self.layers_experts_task2[i](gate_task2_output_final))
            # experts_task2_output = self.layers_experts_task2[i](gate_task2_output_final)
            # experts_task2_output = F.dropout(experts_task2_output, p=0.1)

            # gate shared output
            gate_shared_output = self.layers_experts_shared_gate[i](gate_shared_output_final) # (B, S, C)
            gate_shared_output = F.softmax(gate_shared_output, dim=-1)
            gate_shared_output = torch.matmul(gate_shared_output.unsqueeze(-2), 
                                    torch.cat([experts_task1_output, experts_shared_output, experts_task2_output], dim=-1).permute(0, 1, 3, 2)).squeeze(-2)
            # gate_shared_output_final = gate_shared_output
            gate_shared_output_final = self.layers_layernorm[i][0](gate_shared_output + gate_shared_output_final)

            # gate task1 output
            gate_task1_output = self.layers_experts_task1_gate[i](gate_task1_output_final) # (B, S, C)
            gate_task1_output = F.softmax(gate_task1_output, dim=-1)
            gate_task1_output = torch.matmul(gate_task1_output.unsqueeze(-2), 
                                    torch.cat([experts_task1_output, experts_shared_output], dim=-1).permute(0, 1, 3, 2)).squeeze(-2)
            # gate_task1_output_final = gate_task1_output
            gate_task1_output_final = self.layers_layernorm[i][1](gate_task1_output + gate_task1_output_final)

            # gate task2 output
            gate_task2_output = self.layers_experts_task2_gate[i](gate_task2_output_final) # (B, S, C)
            gate_task2_output = F.softmax(gate_task2_output, dim=-1)
            gate_task2_output = torch.matmul(gate_task2_output.unsqueeze(-2), 
                                    torch.cat([experts_task2_output, experts_shared_output], dim=-1).permute(0, 1, 3, 2)).squeeze(-2)
            # gate_task2_output_final = gate_task2_output
            gate_task2_output_final = self.layers_layernorm[i][2](gate_task2_output + gate_task2_output_final)
        
        return gate_shared_output_final, gate_task1_output_final, gate_task2_output_final



class BILSTM_CRF_Span_Attr_Boundary_PLE_3(BILSTM_CRF_Span_Attr_Boundary_PLE):
    def __init__(self, sequence_encoder, span2id, attr2id, compress_seq=False, share_lstm=False, span_use_lstm=True, attr_use_lstm=False, span_use_crf=True, tagscheme='bmoes', batch_first=True, dropout_rate=0.3, experts_layers=2, experts_num=2):
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
        
        super(BILSTM_CRF_Span_Attr_Boundary_PLE_3, self).__init__(
            sequence_encoder=sequence_encoder,
            span2id=span2id,
            attr2id=attr2id,
            compress_seq=compress_seq,
            share_lstm=share_lstm,
            span_use_lstm=span_use_lstm,
            attr_use_lstm=attr_use_lstm,
            span_use_crf=span_use_crf,
            tagscheme=tagscheme,
            batch_first=batch_first,
            dropout_rate=dropout_rate,
            experts_layers=experts_layers,
            experts_num=experts_num
        )


    def progressive_layered_extraction(self, gate_shared_output_final, gate_task1_output_final, gate_task2_output_final):
        for i in range(self.experts_layers):
            # shared  output
            experts_shared_output = torch.relu(self.layers_experts_shared[i](gate_shared_output_final))
            # experts_shared_output = self.layers_experts_shared[i](gate_shared_output_final)
            experts_shared_output = F.dropout(experts_shared_output, p=0.1)

            # task1 output
            experts_task1_output = torch.relu(self.layers_experts_task1[i](gate_task1_output_final))
            # experts_task1_output = self.layers_experts_task1[i](gate_task1_output_final)
            experts_task1_output = F.dropout(experts_task1_output, p=0.1)

            # task2 output
            experts_task2_output = torch.relu(self.layers_experts_task2[i](gate_task2_output_final))
            # experts_task2_output = self.layers_experts_task2[i](gate_task2_output_final)
            experts_task2_output = F.dropout(experts_task2_output, p=0.1)

            # gate shared output
            gate_shared_output = self.layers_experts_shared_gate[i](gate_shared_output_final) # (B, S, C)
            gate_shared_output = F.softmax(gate_shared_output, dim=-1)
            gate_shared_output = torch.matmul(gate_shared_output.unsqueeze(-2), 
                                    torch.cat([experts_task1_output, experts_shared_output, experts_task2_output], dim=-1).permute(0, 1, 3, 2)).squeeze(-2)
            # gate_shared_output_final = gate_shared_output
            gate_shared_output_final = self.layers_layernorm[i][0](gate_shared_output + gate_shared_output_final)

            # gate task1 output
            gate_task1_output = self.layers_experts_task1_gate[i](gate_task1_output_final) # (B, S, C)
            gate_task1_output = F.softmax(gate_task1_output, dim=-1)
            gate_task1_output = torch.matmul(gate_task1_output.unsqueeze(-2), 
                                    torch.cat([experts_task1_output, experts_shared_output], dim=-1).permute(0, 1, 3, 2)).squeeze(-2)
            # gate_task1_output_final = gate_task1_output
            gate_task1_output_final = self.layers_layernorm[i][1](gate_task1_output + gate_task1_output_final)

            # gate task2 output
            gate_task2_output = self.layers_experts_task2_gate[i](gate_task2_output_final) # (B, S, C)
            gate_task2_output = F.softmax(gate_task2_output, dim=-1)
            gate_task2_output = torch.matmul(gate_task2_output.unsqueeze(-2), 
                                    torch.cat([experts_task2_output, experts_shared_output], dim=-1).permute(0, 1, 3, 2)).squeeze(-2)
            # gate_task2_output_final = gate_task2_output
            gate_task2_output_final = self.layers_layernorm[i][2](gate_task2_output + gate_task2_output_final)
        
        return gate_shared_output_final, gate_task1_output_final, gate_task2_output_final



class BILSTM_CRF_Span_Attr_Three_Boundary_PLE(Base_BILSTM_CRF_Span_Attr_Three):
    def __init__(self, sequence_encoder, span2id, attr2id, compress_seq=False, share_lstm=False, span_use_lstm=True, attr_use_lstm=False, span_use_crf=True, tagscheme='bmoes', batch_first=True, dropout_rate=0.3, experts_layers=2, experts_num=2):
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
        
        super(BILSTM_CRF_Span_Attr_Three_Boundary_PLE, self).__init__(
            sequence_encoder=sequence_encoder,
            span2id=span2id,
            attr2id=attr2id,
            compress_seq=compress_seq,
            share_lstm=share_lstm,
            span_use_lstm=span_use_lstm,
            attr_use_lstm=attr_use_lstm,
            span_use_crf=span_use_crf,
            tagscheme=tagscheme,
            batch_first=batch_first,
            dropout_rate=dropout_rate
        )
        self.experts_layers = experts_layers
        self.experts_num = experts_num
        self.selector_num = 2
        hidden_size = sequence_encoder.hidden_size
        self.layers_experts_shared = nn.ModuleList()
        self.layers_experts_task1 = nn.ModuleList()
        self.layers_experts_task2 = nn.ModuleList()
        self.layers_experts_task3 = nn.ModuleList()
        self.layers_experts_shared_gate = nn.ModuleList()
        self.layers_experts_task1_gate = nn.ModuleList()
        self.layers_experts_task2_gate = nn.ModuleList()
        self.layers_experts_task3_gate = nn.ModuleList()
        for i in range(experts_layers):
            # experts shared
            self.layers_experts_shared.append(Linear3D(hidden_size, hidden_size, experts_num))        

            # experts task1
            self.layers_experts_task1.append(Linear3D(hidden_size, hidden_size, experts_num))            

            # experts task2
            self.layers_experts_task2.append(Linear3D(hidden_size, hidden_size, experts_num))       

            # experts task3
            self.layers_experts_task3.append(Linear3D(hidden_size, hidden_size, experts_num))       

            # gates shared
            self.layers_experts_shared_gate.append(nn.Linear(hidden_size, experts_num * 4))            

            # gate task1
            self.layers_experts_task1_gate.append(nn.Linear(hidden_size, experts_num * self.selector_num))            

            # gate task2
            self.layers_experts_task2_gate.append(nn.Linear(hidden_size, experts_num * self.selector_num))            

            # gate task3
            self.layers_experts_task3_gate.append(nn.Linear(hidden_size, experts_num * self.selector_num)) 


    def progressive_layered_extraction(self, gate_shared_output_final, gate_task1_output_final, gate_task2_output_final, gate_task3_output_final):
        for i in range(self.experts_layers):
            # shared  output
            experts_shared_output = torch.relu(self.layers_experts_shared[i](gate_shared_output_final))
            # experts_shared_output = self.layers_experts_shared[i](gate_shared_output_final)
            
            # task1 output
            experts_task1_output = torch.relu(self.layers_experts_task1[i](gate_task1_output_final))
            # experts_task1_output = self.layers_experts_task1[i](gate_task1_output_final)

            # task2 output
            experts_task2_output = torch.relu(self.layers_experts_task2[i](gate_task2_output_final))
            # experts_task2_output = self.layers_experts_task2[i](gate_task2_output_final)

            # task3 output
            experts_task3_output = torch.relu(self.layers_experts_task3[i](gate_task3_output_final))
            # experts_task3_output = self.layers_experts_task3[i](gate_task3_output_final)

            # gate shared output
            gate_shared_output = self.layers_experts_shared_gate[i](gate_shared_output_final) # (B, S, C)
            gate_shared_output = F.softmax(gate_shared_output, dim=-1)
            gate_shared_output = torch.matmul(gate_shared_output.unsqueeze(-2), 
                                    torch.cat([experts_task1_output, experts_shared_output, experts_task2_output, experts_task3_output], dim=-1).permute(0, 1, 3, 2)).squeeze(-2)
            gate_shared_output_final = gate_shared_output

            # gate task1 output
            gate_task1_output = self.layers_experts_task1_gate[i](gate_task1_output_final) # (B, S, C)
            gate_task1_output = F.softmax(gate_task1_output, dim=-1)
            gate_task1_output = torch.matmul(gate_task1_output.unsqueeze(-2), 
                                    torch.cat([experts_task1_output, experts_shared_output], dim=-1).permute(0, 1, 3, 2)).squeeze(-2)
            gate_task1_output_final = gate_task1_output

            # gate task2 output
            gate_task2_output = self.layers_experts_task2_gate[i](gate_task2_output_final) # (B, S, C)
            gate_task2_output = F.softmax(gate_task2_output, dim=-1)
            gate_task2_output = torch.matmul(gate_task2_output.unsqueeze(-2), 
                                    torch.cat([experts_task2_output, experts_shared_output], dim=-1).permute(0, 1, 3, 2)).squeeze(-2)
            gate_task2_output_final = gate_task2_output

            # gate task3 output
            gate_task3_output = self.layers_experts_task3_gate[i](gate_task3_output_final) # (B, S, C)
            gate_task3_output = F.softmax(gate_task3_output, dim=-1)
            gate_task3_output = torch.matmul(gate_task3_output.unsqueeze(-2), 
                                    torch.cat([experts_task3_output, experts_shared_output], dim=-1).permute(0, 1, 3, 2)).squeeze(-2)
            gate_task3_output_final = gate_task3_output
        
        return gate_shared_output_final, gate_task1_output_final, gate_task2_output_final, gate_task2_output_final


    def forward(self, *args):
        span_seqs_hiddens, attr_seqs_hiddens_start, attr_seqs_hiddens_end = super(BILSTM_CRF_Span_Attr_Three_Boundary_PLE, self).forward(*args)
        _, span_seqs_hiddens, attr_seqs_hiddens_start, attr_seqs_hiddens_end = self.progressive_layered_extraction(self.encoder_output, span_seqs_hiddens, attr_seqs_hiddens_start, attr_seqs_hiddens_end)
        # dropout layer
        span_seqs_hiddens = self.dropout(span_seqs_hiddens)
        attr_seqs_hiddens_start = self.dropout(attr_seqs_hiddens_start)
        attr_seqs_hiddens_end = self.dropout(attr_seqs_hiddens_end)
        # output layer
        logits_span = self.mlp_span(span_seqs_hiddens) # B, S, V
        logits_attr_start = self.mlp_attr_start(attr_seqs_hiddens_start) # B, S, V
        logits_attr_end = self.mlp_attr_end(attr_seqs_hiddens_end) # B, S, V
        
        return logits_span, logits_attr_start, logits_attr_end



class BILSTM_CRF_Span_Attr_Boundary_Sequence_PLE(BILSTM_CRF_Span_Attr_Boundary_PLE):
    def __init__(self, sequence_encoder, span2id, attr2id, compress_seq=False, share_lstm=False, span_use_lstm=True, attr_use_lstm=False, span_use_crf=True, tagscheme='bmoes', batch_first=True, dropout_rate=0.3, experts_layers=2, experts_num=2):
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
        
        super(BILSTM_CRF_Span_Attr_Boundary_Sequence_PLE, self).__init__(
            sequence_encoder=sequence_encoder,
            span2id=span2id,
            attr2id=attr2id,
            compress_seq=compress_seq,
            share_lstm=share_lstm,
            span_use_lstm=span_use_lstm,
            attr_use_lstm=attr_use_lstm,
            span_use_crf=span_use_crf,
            tagscheme=tagscheme,
            batch_first=batch_first,
            dropout_rate=dropout_rate,
            experts_layers=experts_layers,
            experts_num=2
        )
        hidden_size = sequence_encoder.hidden_size
        sequence_size = sequence_encoder.max_length
        self.layers_experts_shared = nn.ModuleList()
        self.layers_experts_task1 = nn.ModuleList()
        self.layers_experts_task2 = nn.ModuleList()
        self.layers_experts_shared_gate = nn.ModuleList()
        self.layers_experts_task1_gate = nn.ModuleList()
        self.layers_experts_task2_gate = nn.ModuleList()
        for i in range(experts_layers):
            # experts shared
            self.layers_experts_shared.append(LinearSequence(sequence_size, hidden_size, hidden_size, experts_num))        

            # experts task1
            self.layers_experts_task1.append(LinearSequence(sequence_size, hidden_size, hidden_size, experts_num))            

            # experts task2
            self.layers_experts_task2.append(LinearSequence(sequence_size, hidden_size, hidden_size, experts_num))       

            # gates shared
            self.layers_experts_shared_gate.append(LinearSequence(sequence_size, hidden_size, experts_num * 3))            

            # gate task1
            self.layers_experts_task1_gate.append(LinearSequence(sequence_size, hidden_size, experts_num * self.selector_num))            

            # gate task2
            self.layers_experts_task2_gate.append(LinearSequence(sequence_size, hidden_size, experts_num * self.selector_num))            



class BILSTM_CRF_Span_Attr_Three_Boundary_Sequence_PLE(BILSTM_CRF_Span_Attr_Three_Boundary_PLE):
    def __init__(self, sequence_encoder, span2id, attr2id, compress_seq=False, share_lstm=False, span_use_lstm=True, attr_use_lstm=False, span_use_crf=True, tagscheme='bmoes', batch_first=True, dropout_rate=0.3, experts_layers=2, experts_num=2):
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
        
        super(BILSTM_CRF_Span_Attr_Three_Boundary_Sequence_PLE, self).__init__(
            sequence_encoder=sequence_encoder,
            span2id=span2id,
            attr2id=attr2id,
            compress_seq=compress_seq,
            share_lstm=share_lstm,
            span_use_lstm=span_use_lstm,
            attr_use_lstm=attr_use_lstm,
            span_use_crf=span_use_crf,
            tagscheme=tagscheme,
            batch_first=batch_first,
            dropout_rate=dropout_rate,
            experts_layers=experts_layers,
            experts_num=2
        )
        hidden_size = sequence_encoder.hidden_size
        sequence_size = sequence_encoder.max_length
        self.layers_experts_shared = nn.ModuleList()
        self.layers_experts_task1 = nn.ModuleList()
        self.layers_experts_task2 = nn.ModuleList()
        self.layers_experts_task3 = nn.ModuleList()
        self.layers_experts_shared_gate = nn.ModuleList()
        self.layers_experts_task1_gate = nn.ModuleList()
        self.layers_experts_task2_gate = nn.ModuleList()
        self.layers_experts_task3_gate = nn.ModuleList()
        for i in range(experts_layers):
            # experts shared
            self.layers_experts_shared.append(LinearSequence(sequence_size, hidden_size, hidden_size, experts_num))        

            # experts task1
            self.layers_experts_task1.append(LinearSequence(sequence_size, hidden_size, hidden_size, experts_num))            

            # experts task2
            self.layers_experts_task2.append(LinearSequence(sequence_size, hidden_size, hidden_size, experts_num))       

            # experts task3
            self.layers_experts_task3.append(LinearSequence(sequence_size, hidden_size, hidden_size, experts_num))       

            # gates shared
            self.layers_experts_shared_gate.append(LinearSequence(sequence_size, hidden_size, experts_num * 4))            

            # gate task1
            self.layers_experts_task1_gate.append(LinearSequence(sequence_size, hidden_size, experts_num * self.selector_num))            

            # gate task2
            self.layers_experts_task2_gate.append(LinearSequence(sequence_size, hidden_size, experts_num * self.selector_num))            

            # gate task3
            self.layers_experts_task3_gate.append(LinearSequence(sequence_size, hidden_size, experts_num * self.selector_num))
