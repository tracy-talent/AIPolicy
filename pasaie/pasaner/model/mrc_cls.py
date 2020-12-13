"""
 Author: liujian
 Date: 2020-12-05 18:07:21
 Last Modified by: liujian
 Last Modified time: 2020-12-05 18:07:21
"""

from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ...utils.entity_extract import extract_kvpairs_by_start_end


class MRC_Span_Pos_CLS(nn.Module):
    def __init__(self, sequence_encoder, tag2id, use_lstm=False, compress_seq=False, add_span_loss=False, dropout_rate=0.1):
        """
        Args:
            sequence_encoder (nn.Module): encoder of sequence
            tag2id (dict): map from tag to id
            use_lstm (bool, optional): whether add lstm layer. Defaults to False.
            compress_seq (bool, optional): whether compress sequence for lstm. Defaults to True.
            dropout_rate (float, optional): dropout rate. Defaults to 0.1.
        """
        super(MRC_Span_Pos_CLS, self).__init__()
        self.sequence_encoder = sequence_encoder
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
        self.start_fc = nn.Linear(self.sequence_encoder.hidden_size, 2)    
        self.end_fc = nn.Linear(self.sequence_encoder.hidden_size, 2)
        if add_span_loss:
            self.span_fc = nn.Linear(self.sequence_encoder.hidden_size * 2, 1)        
        else:
            self.span_fc = None

    def infer(self, text, query_dict):
        """model inference
        Args:
            text (str or list): tokens list or sentence string
            query_dict (dict): map from entity type to MRC query statement

        Returns:
            pos_attr_entities (list[tuple]): list of (pos, entity_attr, entity)
        """
        self.eval()
        negid = -1
        if 'null' in self.tag2id:
            negid = self.tag2id['null']
        if negid == -1:
            raise Exception("negative tag not in 'null'")
        if isinstance(text, str):
            text = self.sequence_encoder.tokenizer.tokenize(text)
        pos_attr_entities = []
        for ent_type in query_dict:
            text_copy = deepcopy(text)
            seqs = list(self.sequence_encoder.tokenize(text_copy, query_dict[ent_type]))
            # if list(self.sequence_encoder.parameters())[0].device.type.startswith('cuda'):
            #     for i in range(len(seqs)):
            #         seqs[i] = seqs[i].cuda()
            valid_len = seqs[-2].sum().item()
            seq_len = seqs[-1].sum().item()
            args = seqs[:-2] + seqs[-1:]
            seq_out, start_logits, end_logits = self.forward(*args)
            spos, tpos = seq_len - valid_len - 1, seq_len - 1
            # construct span logits and span labels for span loss
            if self.span_fc is not None:
                span_out = []
                span_range = []
                for j in range(spos, tpos):
                    if start_logits[0][j][1] <= start_logits[0][j][0]:
                        continue
                    for k in range(spos, tpos):
                        if end_logits[0][k][1] > end_logits[0][k][0]:
                            span_out.append(torch.cat([seq_out[0][j], seq_out[0][k]]))
                            span_range.append((j - spos, k + 1 - spos))
                if len(span_out) > 0:
                    span_out = torch.stack(span_out, dim=0)
                    span_logits = self.span_fc(span_out).squeeze()
                    span_preds = (F.sigmoid(span_logits) >= 0.5).long()
                    pos_attr_entities.extend([(span_range[j], ent_type, ''.join(text[span_range[j][0]:span_range[j][1]])) for j in range(len(span_range)) if span_preds[j]])
            else:
                start_preds = start_logits[:,spos:tpos,:].argmax(dim=-1).squeeze(dim=0).detach().cpu().numpy()
                end_preds = end_logits[:,spos:tpos,:].argmax(dim=-1).squeeze(dim=0).detach().cpu().numpy()
                start_preds_seq = [ent_type if tid else 'null' for tid in start_preds]
                end_preds_seq = [ent_type if tid else 'null' for tid in end_preds]
                pos_attr_entities.extend(extract_kvpairs_by_start_end(start_preds_seq, end_preds_seq, text, self.id2tag[negid]))
            
        return text, pos_attr_entities


    def forward(self, *args):
        """
        Args:
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

        if self.bilstm is not None:
            seq_out = self.dropout(seq_out)
        start_logits = self.start_fc(seq_out)
        end_logits = self.end_fc(seq_out)

        return seq_out, start_logits, end_logits