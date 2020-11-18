"""
 Author: liujian 
 Date: 2020-10-25 20:49:53 
 Last Modified by: liujian 
 Last Modified time: 2020-10-25 20:49:53 
"""

import numpy as np
import logging
from functools import partial
import gc

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
    def __init__(self, path, tag2id, tokenizer, **kwargs):
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
        self.corpus = [] # [[[seq_tokens], [seq_tags]], ..]
        with open(path, 'r', encoding='utf-8') as f:
            tokens = []
            for line in f.readlines():
                line = line.strip().split()
                if len(line) > 0:
                    tokens.append(line)
                elif len(tokens) > 0:
                    self.corpus.append([list(seq) for seq in zip(*tokens)])
                    tokens = []
        self.construct_data()

        self.weight = np.ones((len(self.tag2id)), dtype=np.float32)
        for item in self.corpus:
            for tag in item[1]:
                self.weight[self.tag2id[tag]] += 1.0
        self.weight = 1.0 / (self.weight ** 0.05)
        self.weight = torch.from_numpy(self.weight)

        logging.info("Loaded sentence NER dataset {} with {} lines and {} entity types.".format(path, len(self.data), len(self.tag2id)))

    def construct_data(self):
        self.data = []
        for index in range(len(self.corpus)):
            items = self.corpus[index] # item = [[seq_tokens..], [seq_tags..]]
            seqs = list(self.tokenizer(*items, **self.kwargs))
            
            length = seqs[0].size(1)
            if length >= len(items[1]):
                labels = [self.tag2id[tag] for tag in items[1]]
                labels.extend([self.tag2id['O']] * (length - len(items[1])))
            else:
                labels = [self.tag2id[tag] for tag in items[1][:length]]
                labels[-1] = self.tag2id['O']

            item = [torch.tensor([labels])] + seqs # make labels size (1, L)
            self.data.append(item)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

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
    dataset = SingleNERDataset(path=path, tag2id=tag2id, tokenizer=tokenizer, **kwargs)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=partial(collate_fn, compress_seq))
    return data_loader


class MultiNERDataset(data.Dataset):
    """
    named entity recognition dataset for XLNet MultiTask
    """
    def __init__(self, path, span2id, attr2id, tokenizer, **kwargs):
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
        self.corpus = [] # [[[seqs_token], [seqs_span], [seqs_attr]], ..]
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
                    self.corpus.append([list(seq) for seq in zip(*tokens)])
                    tokens = []
        self.construct_data()

        self.weight_span = np.ones((len(self.span2id)), dtype=np.float32)
        self.weight_attr = np.ones((len(self.attr2id)), dtype=np.float32)
        for item in self.corpus:
            for i in range(len(item[1])):
                self.weight_span[self.span2id[item[1][i]]] += 1.0
                self.weight_attr[self.attr2id[item[2][i]]] += 1.0
        self.weight_span = 1.0 / (self.weight_span ** 0.05)
        self.weight_span = torch.from_numpy(self.weight_span)
        self.weight_attr = 1.0 / (self.weight_attr ** 0.05)
        self.weight_attr = torch.from_numpy(self.weight_attr)

        logging.info("Loaded sentence NER dataset {} with {} lines and {} entity span types and {} entity attr types.".format(path, len(self.data), len(self.span2id), len(self.attr2id)))
    
    def construct_data(self):
        self.data = []
        for index in range(len(self.corpus)):
            items = self.corpus[index] # item = [[seq_tokens..], [seq_tags..], [seq_attrs]]
            seqs = list(self.tokenizer(*items, **self.kwargs))

            length = seqs[0].size(1)
            if length >= len(items[1]):
                labels_span = [self.span2id[tag] for tag in items[1]]
                labels_span.extend([self.span2id['O']] * (length - len(items[1])))
                labels_attr = [self.attr2id[tag] for tag in items[2]]
                labels_attr.extend([self.attr2id['null']] * (length - len(items[2])))
            else:
                labels_span = [self.span2id[tag] for tag in items[1][:length]]
                labels_span[-1] = self.span2id['O']
                labels_attr = [self.attr2id[tag] for tag in items[2][:length]]
                labels_attr[-1] = self.attr2id['null']
            item = [torch.tensor([labels_span]), torch.tensor([labels_attr])] + seqs
            self.data.append(item)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        items = self.data[index] # item = [[seq_tokens..], [seq_tags..], [seq_attrs]]
        seqs = list(self.tokenizer(*items, **self.kwargs))

        length = seqs[0].size(1)
        if length >= len(items[1]):
            labels_span = [self.span2id[tag] for tag in items[1]]
            labels_span.extend([self.span2id['O']] * (length - len(items[1])))
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
    dataset = MultiNERDataset(path=path, span2id=span2id, attr2id=attr2id, tokenizer=tokenizer, **kwargs)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=partial(collate_fn, compress_seq))
    return data_loader


class XLNetSingleNERDataset(SingleNERDataset):
    """
    named entity recognition dataset for XLNet
    """
    def construct_data(self):
        self.data = []
        for index in range(len(self.corpus)):
            items = self.corpus[index] # item = [[seq_tokens..], [seq_tags..]]
            seqs = list(self.tokenizer(*items, **self.kwargs))
            
            length = seqs[0].size(1)
            if length >= len(items[1]):
                labels = [self.tag2id['O']] * (length - len(items[1]))
                labels.extend([self.tag2id[tag] for tag in items[1]])
            else:
                labels = [self.tag2id[tag] for tag in items[1][:length]]
                labels[-2] = self.tag2id['O']
                labels[-1] = self.tag2id['O']
            item = [torch.tensor([labels])] + seqs # make labels size (1, L)
            self.data.append(item)

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
        shuffle, compress_seq=True, num_workers=8, collate_fn=XLNetSingleNERDataset.collate_fn, **kwargs):
    dataset = XLNetSingleNERDataset(path=path, tag2id=tag2id, tokenizer=tokenizer, **kwargs)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=partial(collate_fn, compress_seq))
    return data_loader


class XLNetMultiNERDataset(MultiNERDataset):
    """
    named entity recognition dataset for XLNet MultiTask
    """
    def construct_data(self):
        self.data = []
        for index in range(len(self.corpus)):
            items = self.corpus[index] # item = [[seq_tokens..], [seq_tags..], [seq_attrs]]
            seqs = list(self.tokenizer(*items, **self.kwargs))

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
            item = [torch.tensor([labels_span]), torch.tensor([labels_attr])] + seqs
            self.data.append(item)
    
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
    
    
def XLNetMultiNERDataLoader(path, span2id, attr2id, tokenizer, batch_size, 
        shuffle, compress_seq=True, num_workers=8, collate_fn=XLNetMultiNERDataset.collate_fn, **kwargs):
    dataset = XLNetMultiNERDataset(path=path, span2id=span2id, attr2id=attr2id, tokenizer=tokenizer, **kwargs)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=partial(collate_fn, compress_seq))
    return data_loader


class SpanSingleNERDataset(data.Dataset):
    """
    named entity recognition dataset
    """
    def __init__(self, path, tag2id, encoder, max_span=7, compress_seq=True, **kwargs):
        """
        Args:
            path: path of the input file
            tag2id: dictionary of entity_tag->id mapping
            tokenizer: function of tokenizing
        """
        super().__init__()
        self.path = path
        self.tokenizer = encoder.tokenize
        self.encoder = encoder
        if torch.cuda.is_available():
            self.encoder.cuda()
        self.tag2id = tag2id
        self.max_span = max_span
        self.compress_seq = compress_seq
        self.kwargs = kwargs
        self.data = []
        self.skip_cls, self.skip_sep = 0, 0
        if 'bert' in self.encoder.__class__.__name__.lower():
            self.skip_cls, self.skip_seq = 1, 1
        if 'null' not in self.tag2id:
            raise Exception(f"negative tag is not null")
        self.refresh()

    def refresh(self):
        # Load the file
        if len(self.data) > 0:
            del self.data
            gc.collect()
        self.data = [] # [[label, span_start_out, span_end_out, start_end_pos], ..]
        with open(self.path, 'r', encoding='utf-8') as f:
            tokens = []
            for line in f.readlines():
                line = line.strip().split()
                if len(line) > 0:
                    tokens.append(line)
                elif len(tokens) > 0:
                    items = [list(seq) for seq in zip(*tokens)]
                    seqs = list(self.tokenizer(*items, **self.kwargs)) # (index_tokens:(1,S), att_maks(1,S))
                    if self.compress_seq:
                        seq_len = seqs[-1].sum(dim=-1)
                        for i in range(len(seqs)):
                            if len(seqs[i].size()) > 1 and seqs[i].size(1) > 1:
                                seqs[i] = compress_sequence(seqs[i], seq_len)
                    if torch.cuda.is_available():
                        for i in range(len(seqs)):
                            seqs[i] = seqs[i].cuda()
                    seq_out = self.encoder(*seqs).squeeze(0)
                    seq_len = min(seqs[0].size(1), seqs[-1].sum().item())
                    ub = min(self.max_span, seq_len - self.skip_cls - self.skip_sep)
                    labels = []
                    span_start, span_end = [], []
                    for i in range(2, ub + 1):
                        for j in range(self.skip_cls, seq_len - i + 1 - self.skip_sep):
                            flag = True
                            if items[1][j][0] == 'B' and items[1][j + i - 1][0] == 'E' and items[1][j][2:] == items[1][j + i - 1][2:]:
                                for k in range(j + 1, j + i - 1):
                                    if items[1][k][0] != 'M':
                                        flag = False
                                        break
                            else:
                                flag = False
                            if flag:
                                labels.append(self.tag2id[items[1][j][2:]])
                                # print(''.join(items[0][j:j+i]), items[1][j][2:])
                            else:
                                labels.append(self.tag2id['null'])
                            span_start.append([j])
                            span_end.append([j + i - 1])
                    span_start = torch.tensor(span_start) # (B, 1)
                    span_end = torch.tensor(span_end) # (B, 1)
                    onehot_start = torch.zeros(len(labels), seq_out.size(0)) # (B, S)
                    onehot_end = torch.zeros(len(labels), seq_out.size(0)) # (B, S)
                    onehot_start = onehot_start.scatter_(dim=1, index=span_start, value=1).to(seq_out.device)
                    onehot_end = onehot_end.scatter_(dim=1, index=span_end, value=1).to(seq_out.device)
                    span_start_out = torch.matmul(onehot_start, seq_out).detach().cpu()
                    span_end_out = torch.matmul(onehot_end, seq_out).detach().cpu()
                    for i in range(len(labels)):
                        self.data.append([torch.tensor([labels[i]]), span_start_out[i:i+1], span_end_out[i:i+1], torch.tensor([[span_start[i], span_end[i]]])])
                    tokens = []

        self.weight = np.ones((len(self.tag2id)), dtype=np.float32)
        for item in self.data:
            self.weight[item[0]] += 1.0
        self.weight = 1.0 / (self.weight ** 0.05)
        self.weight = torch.from_numpy(self.weight)

        logging.info("Loaded sentence NER dataset {} with {} lines and {} entity types.".format(self.path, len(self.data), len(self.tag2id)))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index] # [label, entity_pos_pair, indexed_tokens, att_mask]

    @classmethod
    def collate_fn(cls, data):
        seqs = list(zip(*data))
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], dim=0)
        return seqs
    
    
def SpanSingleNERDataLoader(path, tag2id, encoder, batch_size, shuffle, 
                    compress_seq=True, max_span=7, num_workers=8, collate_fn=SpanSingleNERDataset.collate_fn, sampler=None, **kwargs):
    if sampler:
        shuffle = False
    dataset = SpanSingleNERDataset(path=path, tag2id=tag2id, encoder=encoder, max_span=max_span, compress_seq=compress_seq, **kwargs)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            sampler=sampler)
    return data_loader


class SpanMultiNERDataset(data.Dataset):
    """
    named entity recognition dataset
    """
    def __init__(self, path, tag2id, tokenizer, **kwargs):
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
        if 'null' not in self.tag2id:
            raise Exception(f"negative tag not is null")

        # Load the file
        self.corpus = [] # [[[seq_tokens], [seq_tags]], ..]
        with open(path, 'r', encoding='utf-8') as f:
            tokens = []
            for line in f.readlines():
                line = line.strip().split()
                if len(line) > 0:
                    tokens.append(line)
                elif len(tokens) > 0:
                    self.corpus.append([list(seq) for seq in zip(*tokens)])
                    tokens = []
        self.construct_data()

        self.weight = np.ones((len(self.tag2id)), dtype=np.float32)
        for item in self.corpus:
            for tag in item[1]:
                if tag == 'O':
                    self.weight[self.tag2id['null']] += 1.0
                else:
                    self.weight[self.tag2id[tag[2:]]] += 1.0
        self.weight = 1.0 / (self.weight ** 0.05)
        self.weight = torch.from_numpy(self.weight)

        logging.info("Loaded sentence NER dataset {} with {} lines and {} entity types.".format(path, len(self.data), len(self.tag2id)))
    
    def construct_data(self):
        self.data = []
        for index in range(len(self.corpus)):
            items = self.corpus[index] # item = [[seq_tokens..], [seq_tags..]]
            seqs = list(self.tokenizer(*items, **self.kwargs))
            
            length = seqs[0].size(1)
            if length >= len(items[1]):
                start_labels = [self.tag2id[tag[2:]] if tag[0] == 'B' else self.tag2id['null'] for tag in items[1]]
                end_labels = [self.tag2id[tag[2:]] if tag[0] == 'E' else self.tag2id['null'] for tag in items[1]]
                start_labels.extend([self.tag2id['null']] * (length - len(items[1])))
                end_labels.extend([self.tag2id['null']] * (length - len(items[1])))
            else:
                start_labels = [self.tag2id[tag[2:]] if tag[0] == 'B' else self.tag2id['null'] for tag in items[1][:length]]
                end_labels = [self.tag2id[tag[2:]] if tag[0] == 'E' else self.tag2id['null'] for tag in items[1][:length]]
                start_labels[-1] = self.tag2id['null']
                end_labels[-1] = self.tag2id['null']
            item = [torch.tensor([start_labels]), torch.tensor([end_labels])] + seqs # make labels size (1, L)
            self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @classmethod
    def collate_fn(cls, compress_seq, data):
        seqs = list(zip(*data))
        if compress_seq:
            seqs_len = torch.cat(seqs[-1], dim=0).sum(dim=-1)
            sorted_length_indices = seqs_len.argsort(descending=True) 
            seqs_len = seqs_len[sorted_length_indices]
            for i in range(len(seqs)):
                seqs[i] = torch.cat(seqs[i], dim=0)
                if len(seqs[i].size()) > 1 and seqs[i].size(1) > 1:
                    seqs[i] = compress_sequence(seqs[i][sorted_length_indices], seqs_len)
                else:
                    seqs[i] = seqs[i][sorted_length_indices]
        else:
            for i in range(len(seqs)):
                seqs[i] = torch.cat(seqs[i], dim=0)
        return seqs
    
    
def SpanMultiNERDataLoader(path, tag2id, tokenizer, batch_size, 
        shuffle, compress_seq=True, num_workers=8, collate_fn=SpanMultiNERDataset.collate_fn, sampler=None, **kwargs):
    if sampler:
        shuffle = False
    dataset = SpanMultiNERDataset(path=path, tag2id=tag2id, tokenizer=tokenizer, **kwargs)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=partial(collate_fn, compress_seq),
            sampler=sampler)
    return data_loader
