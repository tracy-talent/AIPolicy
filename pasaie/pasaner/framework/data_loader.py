"""
 Author: liujian 
 Date: 2020-10-25 20:49:53 
 Last Modified by: liujian 
 Last Modified time: 2020-10-25 20:49:53 
"""

import numpy as np
import logging
from functools import partial

import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def compress_sequence(seqs, lengths):
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
    def __init__(self, path, tag2id, tokenizer, kwargs):
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
        items = self.data[index] # item = [[seq_tokens..], [seq_tags..]]
        seqs = list(self.tokenizer(*items, **self.kwargs))
        
        length = seqs[0].size(1)
        if length >= len(items[1]):
            labels = [self.tag2id[tag] for tag in items[1]]
            labels.extend([self.tag2id['O']] * (length - len(items[1])))
        else:
            labels = [self.tag2id[tag] for tag in items[1][:length]]
            labels[-1] = self.tag2id['O']

        res = [torch.tensor([labels])] + seqs # make labels size (1, L)
        return res # label, seq1, seq2, ...

    @classmethod
    def collate_fn(cls, compress_seq, data):
        seqs = list(zip(*data))
        if compress_seq:
            seqs_len = torch.cat(seqs[-1], dim=0).sum(dim=-1)
            sorted_length_indices = seqs_len.argsort(descending=True) 
            seqs_len = seqs_len[sorted_length_indices]
            for i in range(len(seqs)):
                seqs[i] = torch.cat(seqs[i], dim=0)
                if seqs[i].size(1) > 1:
                    seqs[i] = compress_sequence(seqs[i][sorted_length_indices], seqs_len)
                else:
                    seqs[i] = seqs[i][sorted_length_indices]
        else:
            for i in range(len(seqs)):
                seqs[i] = torch.cat(seqs[i], dim=0)
        return seqs
    
    
def SingleNERDataLoader(path, tag2id, tokenizer, batch_size, 
        shuffle, compress_seq=True, num_workers=8, collate_fn=SingleNERDataset.collate_fn, **kwargs):
    dataset = SingleNERDataset(path=path, tag2id=tag2id, tokenizer=tokenizer, kwargs=kwargs)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=partial(collate_fn, compress_seq))
    return data_loader


class MultiNERDataset(data.Dataset):
    """
    named entity recognition dataset
    """
    def __init__(self, path, span2id, attr2id, tokenizer, kwargs):
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
        self.kwargs = kwargs
        if 'null' not in self.attr2id:
            raise Exception(f"negative attribute name not is null")

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
        items = self.data[index] # item = [[seq_tokens..], [seq_tags..], [seq_attrs]]
        seqs = list(self.tokenizer(*items, **self.kwargs))

        length = seqs[0].size(1)
        if length >= len(items[1]):
            labels_span = [self.span2id[tag] for tag in items[1]]
            labels_span.extend([self.tag2id['O']] * (length - len(items[1])))
            labels_attr = [self.attr2id[tag] for tag in items[2]]
            labels_attr.extend([self.attr2id['null']] * (length - len(items[2])))
        else:
            labels_span = [self.span2id[tag] for tag in items[1][:length]]
            labels_span[-1] = self.span2id['O']
            labels_attr = [self.attr2id[tag] for tag in items[2][:length]]
            labels_attr[-1] = self.attr2id['null']
        res = [torch.tensor([labels_span]), torch.tensor([labels_attr])] + seqs
        return res # label, seq1, seq2, ...

    @classmethod
    def collate_fn(cls, compress_seq, data):
        seqs = list(zip(*data))
        if compress_seq:
            seqs_len = torch.cat(seqs[-1], dim=0).sum(dim=-1)
            sorted_length_indices = seqs_len.argsort(descending=True) 
            seqs_len = seqs_len[sorted_length_indices]
            for i in range(len(seqs)):
                seqs[i] = torch.cat(seqs[i], dim=0)
                if seqs[i].size(1) > 1:
                    seqs[i] = compress_sequence(seqs[i][sorted_length_indices], seqs_len)
                else:
                    seqs[i] = seqs[i][sorted_length_indices]
        else:
            for i in range(len(seqs)):
                seqs[i] = torch.cat(seqs[i], dim=0)
        return seqs
    
    
def MultiNERDataLoader(path, span2id, attr2id, tokenizer, batch_size, 
        shuffle, compress_seq=True, num_workers=8, collate_fn=MultiNERDataset.collate_fn, **kwargs):
    dataset = MultiNERDataset(path=path, span2id=span2id, attr2id=attr2id, tokenizer=tokenizer, kwargs=kwargs)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=partial(collate_fn, compress_seq))
    return data_loader


class XLNetSingleNERDataset(data.Dataset):
    """
    named entity recognition dataset
    """
    def __init__(self, path, tag2id, tokenizer, kwargs):
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
        items = self.data[index] # item = [[seq_tokens..], [seq_tags..]]
        seqs = list(self.tokenizer(*items, **self.kwargs))
        
        length = seqs[0].size(1)
        if length >= len(items[1]):
            labels = [self.tag2id['O']] * (length - len(items[1]))
            labels.extend([self.tag2id[tag] for tag in items[1]])
        else:
            labels = [self.tag2id[tag] for tag in items[1][:length]]
            labels[-2] = self.tag2id['O']
            labels[-1] = self.tag2id['O']
        res = [torch.tensor([labels])] + seqs # make labels size (1, L)
        return res # label, seq1, seq2, ...

    @classmethod
    def collate_fn(cls, compress_seq, data):
        seqs = list(zip(*data))
        if compress_seq:
            seqs_len = torch.cat(seqs[-1], dim=0).sum(dim=-1)
            sorted_length_indices = seqs_len.argsort(descending=True) 
            seqs_len = seqs_len[sorted_length_indices]
            for i in range(len(seqs)):
                seqs[i] = torch.cat(seqs[i], dim=0)
                if seqs[i].size(1) > 1:
                    seqs[i] = torch.from_numpy(seqs[i].numpy()[:,::-1])
                    seqs[i] = torch.from_numpy(compress_sequence(seqs[i][sorted_length_indices], seqs_len).numpy()[:,::-1])
                else:
                    seqs[i] = seqs[i][sorted_length_indices]
        else:
            for i in range(len(seqs)):
                seqs[i] = torch.cat(seqs[i], dim=0)
        return seqs
    
    
def XLNetSingleNERDataLoader(path, tag2id, tokenizer, batch_size, 
        shuffle, compress_seq=True, num_workers=16, collate_fn=SingleNERDataset.collate_fn, **kwargs):
    dataset = XLNetSingleNERDataset(path=path, tag2id=tag2id, tokenizer=tokenizer, kwargs=kwargs)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=partial(collate_fn, compress_seq))
    return data_loader


class XLNetMultiNERDataset(data.Dataset):
    """
    named entity recognition dataset
    """
    def __init__(self, path, span2id, attr2id, tokenizer, kwargs):
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
        self.kwargs = kwargs
        if 'null' not in self.attr2id:
            raise Exception(f"negative attribute name not is null")

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
        items = self.data[index] # item = [[seq_tokens..], [seq_tags..], [seq_attrs]]
        seqs = list(self.tokenizer(*items, **self.kwargs))

        length = seqs[0].size(1)
        if length > len(items[1]):
            labels = [self.tag2id['O']] * (length - len(items[1]))
            labels.extend([self.tag2id[tag] for tag in items[1]])
        else:
            labels = [self.tag2id[tag] for tag in items[1][:length]]
            labels[-2] = self.tag2id['O']
            labels[-1] = self.tag2id['O']

        length = seqs[0].size(1)
        if length >= len(items[1]):
            labels_span = [self.span2id['O']] * (length - len(items[1]))
            labels_span.extend([self.span2id[tag] for tag in items[1]])
            labels_attr = [self.attr2id['O']] * (length - len(items[2]))
            labels_attr.extend([self.attr2id[tag] for tag in items[2]])
        else:
            labels_span = [self.span2id[tag] for tag in items[1][:length]]
            labels_span[-2] = self.span2id['O']
            labels_span[-1] = self.span2id['O']
            labels_attr = [self.attr2id[tag] for tag in items[2][:length]]
            labels_attr[-2] = self.attr2id['null']
            labels_attr[-1] = self.attr2id['null']
        res = [torch.tensor([labels_span]), torch.tensor([labels_attr])] + seqs
        return res # label, seq1, seq2, ...

    @classmethod
    def collate_fn(cls, compress_seq, data):
        seqs = list(zip(*data))
        if compress_seq:
            seqs_len = torch.cat(seqs[-1], dim=0).sum(dim=-1)
            sorted_length_indices = seqs_len.argsort(descending=True) 
            seqs_len = seqs_len[sorted_length_indices]
            for i in range(len(seqs)):
                seqs[i] = torch.cat(seqs[i], dim=0)
                if seqs[i].size(1) > 1:
                    seqs[i] = compress_sequence(seqs[i][sorted_length_indices], seqs_len)
                else:
                    seqs[i] = seqs[i][sorted_length_indices]
        else:
            for i in range(len(seqs)):
                seqs[i] = torch.cat(seqs[i], dim=0)
        return seqs
    
    
def XLNetMultiNERDataLoader(path, span2id, attr2id, tokenizer, batch_size, 
        shuffle, compress_seq=True, num_workers=8, collate_fn=MultiNERDataset.collate_fn, **kwargs):
    dataset = XLNetMultiNERDataset(path=path, span2id=span2id, attr2id=attr2id, tokenizer=tokenizer, kwargs=kwargs)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=partial(collate_fn, compress_seq))
    return data_loader