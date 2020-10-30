"""
 Author: liujian 
 Date: 2020-10-25 20:49:53 
 Last Modified by: liujian 
 Last Modified time: 2020-10-25 20:49:53 
"""

import numpy as np
import logging

import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def compress_seq(seqs, lengths):
    """compress padding in batch

    Args:
        seqs (torch.LongTensor->(B, L)): batch seqs, sorted by seq's actual length in decreasing order
        lengths (torch.LongTensor->(B)): length of every seq in batch in decreasing order

    Returns:
        torch.LongTensor: compressed batch seqs
    """
    packed_seqs = pack_padded_sequence(input=seqs, lengths=lengths.detach().cpu().numpy(), batch_first=True)
    seqs, _ = pad_packed_sequence(sequence=packed_seqs, batch_first=True)
    return seqs


class SingleNERDataset(data.Dataset):
    """
    named entity recognition dataset
    """
    def __init__(self, path, tag2id, tokenizer, is_bert_encoder, kwargs):
        """
        Args:
            path: path of the input file
            tag2id: dictionary of entity_tag->id mapping
            tokenizer: function of tokenizing
        """
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.is_bert_encoder = is_bert_encoder
        self.kwargs = kwargs

        # Load the file
        self.data = [] # [[[seq_tokens], [seq_tags]], ..]
        with open(path, 'r', encoding='utf-8') as f:
            tokens = []
            for line in f.readlines():
                line = line.strip().split()
                if len(line) > 0:
                    tokens.append(line)
                elif len(tokens) > 0:
                    self.data.append([list(seq) for seq in zip(*tokens)])
                    tokens = []

        self.weight = np.ones((len(self.tag2id)), dtype=np.float32)
        for item in self.data:
            for tag in item[1]:
                self.weight[self.tag2id[tag]] += 1.0
        self.weight = 1.0 / (self.weight ** 0.05)
        self.weight = torch.from_numpy(self.weight)

        logging.info("Loaded sentence RE dataset {} with {} lines and {} relations.".format(path, len(self.data), len(self.tag2id)))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index] # item = [(seq_tokens..), (seq_tags..)], item[0]和item[1]是tuple
        seqs = list(self.tokenizer(item[0], **self.kwargs))

        if self.is_bert_encoder:
            labels = [self.tag2id['O']] + [self.tag2id[tag] for tag in item[1]] + [self.tag2id['O']] # '[CLS]...[SEP]'
        else:
            labels = [self.tag2id[tag] for tag in item[1]]

        labels.extend([self.tag2id['O']] * (seqs[0].size(1) - len(labels)))
        res = [torch.tensor([labels])] + seqs # make labels size (1, L)
        return res # label, seq1, seq2, ...

    @classmethod
    def collate_fn(cls, data):
        seqs = list(zip(*data))
        seqs_len = torch.cat(seqs[-1], dim=0).sum(dim=-1)
        sorted_length_indices = seqs_len.argsort(descending=True) 
        seqs_len = seqs_len[sorted_length_indices]
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], dim=0)
            if seqs[i].size(1) > 1:
                seqs[i] = compress_seq(seqs[i][sorted_length_indices], seqs_len)
            else:
                seqs[i] = seqs[i][sorted_length_indices]

        return seqs
    
    
def SingleNERDataLoader(path, tag2id, tokenizer, is_bert_encoder, batch_size, 
        shuffle, num_workers=8, collate_fn=SingleNERDataset.collate_fn, **kwargs):
    dataset = SingleNERDataset(path=path, tag2id=tag2id, tokenizer=tokenizer, is_bert_encoder=is_bert_encoder, kwargs=kwargs)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return data_loader


class MultiNERDataset(data.Dataset):
    """
    named entity recognition dataset
    """
    def __init__(self, path, span2id, attr2id, tokenizer, is_bert_encoder, kwargs):
        """
        Args:
            path: path of the input file
            span2id: dictionary of entity_span->id mapping
            attr2id: dictionary of entity_attr->id mapping
            tokenizer: function of tokenizing
        """
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.span2id = span2id
        self.attr2id = attr2id
        self.is_bert_encoder = is_bert_encoder
        self.kwargs = kwargs
        self.neg_attrid = -1
        neg_attrname = ['NA', 'na', 'null', 'Other', 'other']
        for attr_name in neg_attrname:
            if attr_name in self.attr2id:
                self.neg_attrid = self.attr2id[attr_name]
                break
        if self.neg_attrid == -1:
            raise Exception(f"negative attribute name not in {neg_attrname}")

        # Load the file
        self.data = [] # [[[seqs_token], [seqs_span], [seqs_attr]], ..]
        with open(path, 'r', encoding='utf-8') as f:
            tokens = []
            for line in f.readlines():
                line = line.strip().split()
                if len(line) > 0:
                    if line[1][0] == 'O':
                        line.append('null')
                    else:
                        line.append(line[1][2:])
                        line[1] = line[1][0]
                    tokens.append(line)
                elif len(tokens) > 0:
                    self.data.append([list(seq) for seq in zip(*tokens)])
                    tokens = []

        self.weight_span = np.ones((len(self.span2id)), dtype=np.float32)
        self.weight_attr = np.ones((len(self.attr2id)), dtype=np.float32)
        for item in self.data:
            for i in range(len(item[1])):
                self.weight_span[self.span2id[item[1][i]]] += 1.0
                self.weight_attr[self.attr2id[item[2][i]]] += 1.0
        self.weight_span = 1.0 / (self.weight_span ** 0.05)
        self.weight_span = torch.from_numpy(self.weight_span)
        self.weight_attr = 1.0 / (self.weight_attr ** 0.05)
        self.weight_attr = torch.from_numpy(self.weight_attr)

        logging.info("Loaded sentence RE dataset {} with {} lines and {} relations.".format(path, len(self.data), len(self.span2id)))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index] # item = [(seq_tokens..), (seq_tags..)], item[0]和item[1]是tuple
        seqs = list(self.tokenizer(item[0], **self.kwargs))
        if self.is_bert_encoder:
            labels_span = [self.span2id['O']] + [self.span2id[span] for span in item[1]] + [self.span2id['O']] # '[CLS]...[SEP]'
            labels_attr = [self.neg_attrid] + [self.attr2id[attr] for attr in item[2]] + [self.neg_attrid]
        else:
            labels_span = [self.span2id[span] for span in item[1]]
            labels_attr = [self.attr2id[attr] for attr in item[2]]
        labels_span.extend([self.span2id['O']] * (seqs[0].size(1) - len(labels_span)))
        labels_attr.extend([self.neg_attrid] * (seqs[0].size(1) - len(labels_attr)))
        res = [torch.tensor([labels_span]), torch.tensor([labels_attr])] + seqs
        return res # label, seq1, seq2, ...

    @classmethod
    def collate_fn(cls, data):
        seqs = list(zip(*data))
        seqs_len = torch.cat(seqs[-1], dim=0).sum(dim=-1)
        sorted_length_indices = seqs_len.argsort(descending=True) 
        seqs_len = seqs_len[sorted_length_indices]
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], dim=0)
            if seqs[i].size(1) > 1:
                seqs[i] = compress_seq(seqs[i][sorted_length_indices], seqs_len)
            else:
                seqs[i] = seqs[i][sorted_length_indices]

        return seqs
    
    
def MultiNERDataLoader(path, span2id, attr2id, tokenizer, is_bert_encoder, batch_size, 
        shuffle, num_workers=8, collate_fn=MultiNERDataset.collate_fn, **kwargs):
    dataset = MultiNERDataset(path=path, span2id=span2id, attr2id=attr2id, tokenizer=tokenizer, is_bert_encoder=is_bert_encoder, kwargs=kwargs)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return data_loader