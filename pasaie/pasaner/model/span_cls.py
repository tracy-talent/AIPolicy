"""
 Author: liujian 
 Date: 2020-11-16 15:10:30 
 Last Modified by: liujian 
 Last Modified time: 2020-11-16 15:10:30 
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ...module.nn import FeedForwardNetwork, PoolerStartLogits, PoolerEndLogits
from ...utils.entity_extract import extract_kvpairs_by_start_end


class Span_Cat_CLS(nn.Module):
    def __init__(self, sequence_encoder, tag2id, ffn_hidden_size=150, width_embedding_size=150, max_span=7, dropout_rate=0.1):
        """
        Args:
            sequence_encoder (nn.Module): encoder of sequence 
            tag2id (dict): map from tag to id
            ffn_hidden_size (int, optional): hidden size of FeedForwardNetWork. Defaults to 150.
            width_embedding_size (int, optional): embedding size of width embedding. Defaults to 150.
            max_span (int, optional): max length of entity in corpus. Defaults to 7.
            dropout_rate (float, optional): dropout rate. Defaults to 0.1.
        """
        
        super(Span_Cat_CLS, self).__init__()
        self.sequence_encoder = sequence_encoder
        self.max_span = max_span
        self.tag2id = tag2id
        self.id2tag = {}
        for tag, tid in tag2id.items():
            self.id2tag[tid] = tag
        self.ffn = FeedForwardNetwork(self.sequence_encoder.hidden_size * 2 + width_embedding_size, 
                                    ffn_hidden_size, len(self.tag2id), dropout_rate)
        self.width_embedding = nn.Embedding(max_span, width_embedding_size)
        

    def infer(self, text):
        """model inference
        Args:
            text (str or list): tokens list or sentence striDataLoaderng
        
        Returns:
            pos_attr_entities (list[tuple]): list of (pos, entity_attr, entity)
        """
        self.eval()
        negid = -1
        if 'null' in self.tag2id:
            negid = self.tag2id['null']
        if negid == -1:
            raise Exception("negative tag not in 'null'")
        skip_cls, skip_sep = 0, 0
        if 'bert' in self.sequence_encoder.__class__.__name__.lower():
            skip_cls, skip_sep = 1, 1 # contain '[CLS]' and '[SEP]'
        seqs = list(self.sequence_encoder.tokenize(text))
        # if list(self.sequence_encoder.parameters())[0].device.type.startswith('cuda'):
        #     for i in range(len(seqs)):
        #         seqs[i] = seqs[i].cuda()
        seq_out = self.sequence_encoder(*seqs).squeeze(0)
        seq_len = seqs[-1].sum().item()
        ub = min(self.max_span, seq_len - skip_sep - skip_sep)
        span_start, span_end = [], []
        for i in range(2, ub + 1):
            for j in range(skip_cls, seq_len - i + 1 - skip_sep):
                span_start.append([j])
                span_end.append([j + i - 1])
        span_start = torch.tensor(span_start) # (B, 1)
        span_end = torch.tensor(span_end) # (B, 1)
        onehot_start = torch.zeros(len(span_start), seq_out.size(0)) # (B, S)
        onehot_end = torch.zeros(len(span_end), seq_out.size(0)) # (B, S)
        onehot_start = onehot_start.scatter_(dim=1, index=span_start, value=1).to(seq_out.device)
        onehot_end = onehot_end.scatter_(dim=1, index=span_end, value=1).to(seq_out.device)
        span_start_out = torch.matmul(onehot_start, seq_out)
        span_end_out = torch.matmul(onehot_end, seq_out)
        span_pos = torch.cat([span_start, span_end], dim=-1).to(seq_out.device)

        if isinstance(text, str):
            text = self.sequence_encoder.tokenizer.tokenize(text)
        pos_attr_entities = []
        for i in range(0, len(span_pos), 32):
            logits = self.forward(span_start_out[i:i+32], span_end_out[i:i+32], span_pos[i:i+32])
            preds = logits.argmax(dim=-1).cpu().numpy()
            for j in range(len(preds)):
                if preds[j] == negid:
                    continue
                spos, tpos = span_pos[i+j][0].item() - skip_cls, span_pos[i+j][1].item() - skip_cls + 1
                pos_attr_entities.append(((spos, tpos), self.id2tag[preds[j]], ''.join([word[2:] if word.startswith('##') else word for word in text[spos:tpos]])))
            
        return text, pos_attr_entities


    def forward(self, span_start_encoding, span_end_encoding, span_pos):
        """
        Args:
            span_start_encoding (torch.tensor): entity span start encoding, (B, d)
            span_end_encoding (torch.tensor): entity span end encoding, (B, d)
            span_pos (torch.tensor): start and end position of entity, (B, d)

        Returns:
            [type]: [description]
        """
        width_embed = self.width_embedding(span_pos[:, 1] - span_pos[:, 0]) # (B, d)
        logits = self.ffn(torch.cat([span_start_encoding, span_end_encoding, width_embed], dim=-1)) # (B, C)
        return logits


class Span_Pos_CLS(nn.Module):
    def __init__(self, sequence_encoder, tag2id, use_lstm=False, compress_seq=False, soft_label=False, dropout_rate=0.1):
        """
        Args:
            sequence_encoder (nn.Module): encoder of sequence
            tag2id (dict): map from tag to id
            use_lstm (bool, optional): whether add lstm layer. Defaults to False.
            compress_seq (bool, optional): whether compress sequence for lstm. Defaults to True.
            soft_label (bool, optional): use one hot if soft_label is True. Defaults to False.
            dropout_rate (float, optional): dropout rate. Defaults to 0.1.
        """
        super(Span_Pos_CLS, self).__init__()
        self.sequence_encoder = sequence_encoder
        self.soft_label = soft_label
        self.num_labels = len(tag2id)
        self.tag2id = tag2id
        self.id2tag = {}
        for tag, tid in tag2id.items():
            self.id2tag[tid] = tag

        if use_lstm:
            self.bilstm = nn.LSTM(input_size=sequence_encoder.hidden_size, 
                                hidden_size=sequence_encoder.hidden_size, 
                                num_layers=1, 
                                bidirectional=True, 
                                batch_first=True)
        else:
            self.bilstm = None
        self.compress_seq = compress_seq

        self.dropout = nn.Dropout(dropout_rate)
        self.start_fc = PoolerStartLogits(self.sequence_encoder.hidden_size, self.num_labels)
        if self.soft_label:
            self.end_fc = PoolerEndLogits(self.sequence_encoder.hidden_size + self.num_labels, self.num_labels)
        else:
            self.end_fc = PoolerEndLogits(self.sequence_encoder.hidden_size + 1, self.num_labels)
        

    def infer(self, text):
        """model inference
        Args:
            text (str or list): tokens list or sentence string
        
        Returns:
            pos_attr_entities (list[tuple]): list of (pos, entity_attr, entity)
        """
        self.eval()
        negid = -1
        if 'null' in self.tag2id:
            negid = self.tag2id['null']
        if negid == -1:
            raise Exception("negative tag not in 'null'")
        seqs = list(self.sequence_encoder.tokenize(text))
        # if list(self.sequence_encoder.parameters())[0].device.type.startswith('cuda'):
        #     for i in range(len(seqs)):
        #         seqs[i] = seqs[i].cuda()
        seq_len = seqs[-1].sum().item()
        start_logits, end_logits = self.forward(None, *seqs)
        start_preds = start_logits[:,:seq_len,:].argmax(dim=-1).squeeze().detach().cpu().numpy()
        end_preds = end_logits[:,:seq_len,:].argmax(dim=-1).squeeze().detach().cpu().numpy()
        spos, tpos = 0, seq_len
        if 'bert' in self.sequence_encoder.__class__.__name__.lower():
            spos, tpos = 1, -1
        start_preds_seq = [self.id2tag[tid] for tid in start_preds[spos:tpos]]
        end_preds_seq = [self.id2tag[tid] for tid in end_preds[spos:tpos]]
        if isinstance(text, str):
            text = self.sequence_encoder.tokenizer.tokenize(text)
        pos_attr_entities = extract_kvpairs_by_start_end(start_preds_seq, end_preds_seq, text, self.id2tag[negid])
            
        return text, pos_attr_entities


    def forward(self, start_labels=None, *args):
        """
        Args:
            start_labels (torch.tensor, optional): labels of entity span start position, (B, S). Defaults to None.

        Returns:
            start_logits (torch.tensor): model outputs for entity span start position, (B, S, d)
            end_logits (torch.tensor): model outputs for entity span end position, (B, S, d)
        """
        seq_out = self.sequence_encoder(*args)
        if self.bilstm is not None:
            if self.compress_seq:
                att_mask = args[-1]
                seqs_length = att_mask.sum(dim=-1).detach().cpu()
                seqs_rep_packed = pack_padded_sequence(seq_out, seqs_length, batch_first=True)
                seqs_hiddens_packed, _ = self.bilstm(seqs_rep_packed)
                seqs_hiddens, _ = pad_packed_sequence(seqs_hiddens_packed, batch_first=True) # B, S, D
            else:
                seqs_hiddens, _ = self.bilstm(seq_out)
            # seqs_hiddens = nn.functional.dropout(seqs_hiddens, 0.2)
            seq_out = torch.add(*seqs_hiddens.chunk(2, dim=-1))

        seq_out = self.dropout(seq_out)
        start_logits = self.start_fc(seq_out)
        if start_labels is not None and self.training:
            if self.soft_label:
                label_logits = torch.zeros_like(start_logits).to(start_labels.device)
                label_logits.scatter_(2, start_labels.unsqueeze(2), 1)
            else:
                label_logits = start_labels.unsqueeze(2).float()
        else:
            start_preds = start_logits.argmax(dim=-1, keepdims=True)
            label_logits = torch.zeros_like(start_logits).to(start_preds.device)
            label_logits.scatter_(2, start_preds, 1)
            # label_logits = F.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = label_logits.argmax(dim=-1, keepdims=True).float()
        end_logits = self.end_fc(seq_out, label_logits)

        return start_logits, end_logits