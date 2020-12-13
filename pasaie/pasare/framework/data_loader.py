import torch
import torch.utils.data as data
import os, random, json, logging
from functools import partial
from collections import defaultdict
import numpy as np
import sklearn.metrics
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


class SentenceREDataset(data.Dataset):
    """
    Sentence-level relation extraction dataset
    """

    def __init__(self, path, rel2id, tokenizer, **kwargs):
        """
        Args:
            path: path of the input file
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
        """
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.kwargs = kwargs
        # Load the file
        f = open(path)
        self.corpus = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) > 0:
                    self.corpus.append(eval(line))
        self._construct_data()

        self.weight = np.ones((len(self.rel2id)), dtype=np.float32)
        for item in self.corpus:
            self.weight[self.rel2id[item['relation']]] += 1.0
        self.weight = 1.0 / (self.weight ** 0.05)
        self.weight = torch.from_numpy(self.weight)

        logging.info("Loaded sentence RE dataset {} with {} lines and {} relations.".format(path, len(self.data),
                                                                                            len(self.rel2id)))
    def _construct_data(self):
        self.data = []
        for index in range(len(self.corpus)):
            item = self.corpus[index]
            seq = list(self.tokenizer(item, **self.kwargs))
            data_item = [torch.tensor([self.rel2id[item['relation']]])] + seq  # label, seq1, seq2, ...
            self.data.append(data_item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @classmethod
    def collate_fn(cls, compress_seq, data):
        seqs = list(zip(*data))
        if compress_seq:
            seqs_len = torch.cat(seqs[-1], dim=0).sum(dim=-1) # (B)
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


def SentenceRELoader(path, rel2id, tokenizer, batch_size, shuffle, drop_last=False, 
                    compress_seq=True, num_workers=8, collate_fn=SentenceREDataset.collate_fn, sampler=None, **kwargs):
    if sampler:
        shuffle = False
    dataset = SentenceREDataset(path=path, rel2id=rel2id, tokenizer=tokenizer, **kwargs)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  drop_last=drop_last,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=partial(collate_fn, compress_seq),
                                  sampler=sampler)
    return data_loader


class SentenceWithDSPREDataset(SentenceREDataset):
    """
    Sentence-level relation extraction dataset with DSP feature
    """
    def __init__(self, path, rel2id, tokenizer, max_dsp_path_length=-1, is_bert_encoder=True, **kwargs):
        """[summary]

        Args:
            path: path of the input file
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
            max_dsp_path_length (int, optional): max length of DSP path length. Defaults to -1.
            is_bert_encoder (bool, optional): whether encoder is bert. Defaults to True.
        """
        """
        Args:
            path: path of the input file
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
        """
        self.max_dsp_path_length = max_dsp_path_length
        self.is_bert_encoder = is_bert_encoder
        super(SentenceWithDSPREDataset, self).__init__(path, rel2id, tokenizer, **kwargs)

    def _construct_data(self):
        if self.max_dsp_path_length > 0:
            self.dsp_path = []
            dsp_path = [self.path[:-4] + '_ddp_dsp_path.txt' for datasp in ['train', 'val', 'test'] if datasp in self.path][0]
            with open(dsp_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = eval(line.strip())
                    ent_h_path = line['ent_h_path']
                    ent_t_path = line['ent_t_path']
                    ent_h_length = len(line['ent_h_path'])
                    ent_t_length = len(line['ent_t_path'])
                    if ent_h_length < self.max_dsp_path_length:
                        while len(ent_h_path) < self.max_dsp_path_length:
                            ent_h_path.append(0)
                    ent_h_path = ent_h_path[:self.max_dsp_path_length]
                    if ent_t_length < self.max_dsp_path_length:
                        while len(ent_t_path) < self.max_dsp_path_length:
                            ent_t_path.append(0)
                    ent_t_path = ent_t_path[:self.max_dsp_path_length]
                    ent_h_path = torch.tensor([ent_h_path]).long() + (1 if self.is_bert_encoder else 0)
                    ent_t_path = torch.tensor([ent_t_path]).long() + (1 if self.is_bert_encoder else 0)
                    ent_h_length = torch.tensor([min(ent_h_length, self.max_dsp_path_length)]).long()
                    ent_t_length = torch.tensor([min(ent_t_length, self.max_dsp_path_length)]).long()
                    self.dsp_path.append([ent_h_path, ent_t_path, ent_h_length, ent_t_length])

        self.data = []
        for index in range(len(self.corpus)):
            item = self.corpus[index]
            seq = list(self.tokenizer(item, **self.kwargs))
            data_item = [torch.tensor([self.rel2id[item['relation']]])] + seq  # label, seq1, seq2, ...
            if self.max_dsp_path_length > 0:
                data_item += self.dsp_path[index]
            self.data.append(data_item)
            if (index + 1) % 500 == 0:
                logging.info(f'parsed {index + 1} sentences for DSP path')

    @classmethod
    def collate_fn(cls, compress_seq, data):
        seqs = list(zip(*data))
        if compress_seq:
            seqs_len = torch.cat(seqs[-5], dim=0).sum(dim=-1) # (B)
            sorted_length_indices = seqs_len.argsort(descending=True) 
            seqs_len = seqs_len[sorted_length_indices]
            for i in range(len(seqs)):
                seqs[i] = torch.cat(seqs[i], dim=0)
                if i < len(seqs) - 4 and len(seqs[i].size()) > 1 and seqs[i].size(1) > 1:
                    seqs[i] = compress_sequence(seqs[i][sorted_length_indices], seqs_len)
                else:
                    seqs[i] = seqs[i][sorted_length_indices]
        else:
            for i in range(len(seqs)):
                seqs[i] = torch.cat(seqs[i], dim=0)

        return seqs


def SentenceWithDSPRELoader(path, rel2id, tokenizer, batch_size, shuffle, drop_last=False, compress_seq=True, max_dsp_path_length=-1, 
                            is_bert_encoder=True, num_workers=0, collate_fn=SentenceWithDSPREDataset.collate_fn, sampler=None, **kwargs):
    if sampler:
        shuffle = False
    dataset = SentenceWithDSPREDataset(path=path, rel2id=rel2id, tokenizer=tokenizer, 
                max_dsp_path_length=max_dsp_path_length, is_bert_encoder=is_bert_encoder, **kwargs)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  drop_last=drop_last,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=partial(collate_fn, compress_seq),
                                  sampler=sampler)
    return data_loader


class BagREDataset(data.Dataset):
    """
    Bag-level relation extraction dataset. Note that relation of NA should be named as 'NA'.
    """

    def __init__(self, path, rel2id, tokenizer, entpair_as_bag=False, bag_size=0, mode=None):
        """
        Args:
            path: path of the input file
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
            entpair_as_bag: if True, bags are constructed based on same
                entity pairs instead of same relation facts (ignoring 
                relation labels)
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.entpair_as_bag = entpair_as_bag
        self.bag_size = bag_size

        # Load the file
        f = open(path)
        self.data = []
        for line in f:
            line = line.rstrip()
            if len(line) > 0:
                self.data.append(eval(line))
        f.close()

        # Construct bag-level dataset (a bag contains instances sharing the same relation fact)
        if mode == None:
            self.weight = np.ones((len(self.rel2id)), dtype=np.float32)
            self.bag_scope = []
            self.name2id = {}
            self.bag_name = []
            self.facts = {}
            for idx, item in enumerate(self.data):
                fact = (item['h']['id'], item['t']['id'], item['relation'])
                if item['relation'] != 'NA':
                    self.facts[fact] = 1
                if entpair_as_bag:
                    name = (item['h']['id'], item['t']['id'])
                else:
                    name = fact
                if name not in self.name2id:
                    self.name2id[name] = len(self.name2id)
                    self.bag_scope.append([])
                    self.bag_name.append(name)
                self.bag_scope[self.name2id[name]].append(idx)
                self.weight[self.rel2id[item['relation']]] += 1.0
            self.weight = 1.0 / (self.weight ** 0.05)
            self.weight = torch.from_numpy(self.weight)
        else:
            pass

    def __len__(self):
        return len(self.bag_scope)

    def __getitem__(self, index):
        bag = self.bag_scope[index]
        if self.bag_size > 0:
            if self.bag_size <= len(bag):
                resize_bag = random.sample(bag, self.bag_size)
            else:
                resize_bag = bag + list(np.random.choice(bag, self.bag_size - len(bag)))
            bag = resize_bag

        seqs = None
        rel = self.rel2id[self.data[bag[0]]['relation']]
        for sent_id in bag:
            item = self.data[sent_id]
            seq = list(self.tokenizer(item))
            if seqs is None:
                seqs = []
                for i in range(len(seq)):
                    seqs.append([])
            for i in range(len(seq)):
                seqs[i].append(seq[i])
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], 0)  # (n, L), n is the size of bag
        return [rel, self.bag_name[index], len(bag)] + seqs

    @classmethod
    def collate_fn(cls, data):
        data = list(zip(*data))
        label, bag_name, count = data[:3]
        seqs = data[3:]
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], 0)  # (sumn, L)
            seqs[i] = seqs[i].unsqueeze(0)
            # seqs[i] = seqs[i].expand((torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1, ) + seqs[i].size())
        scope = []  # (B, 2)
        start = 0
        for c in count:
            scope.append((start, start + c))
            start += c
        assert (start == seqs[0].size(1))
        scope = torch.tensor(scope).long()
        label = torch.tensor(label).long()  # (B)
        return [label, bag_name, scope] + seqs

    @classmethod
    def collate_bag_size_fn(cls, data):
        data = list(zip(*data))
        label, bag_name, count = data[:3]
        seqs = data[3:]
        for i in range(len(seqs)):
            seqs[i] = torch.stack(seqs[i], 0)  # (batch, bag, L)
        scope = []  # (B, 2)
        start = 0
        for c in count:
            scope.append((start, start + c))
            start += c
        label = torch.tensor(label).long()  # (B)
        return [label, bag_name, scope] + seqs

    def eval(self, pred_result):
        """
        Args:
            pred_result: a list with dict {'entpair': (head_id, tail_id), 'relation': rel, 'score': score}.
                Note that relation of NA should be excluded.
        Return:
            {'prec': narray[...], 'rec': narray[...], 'mean_prec': xx, 'f1': xx, 'auc': xx}
                prec (precision) and rec (recall) are in micro style.
                prec (precision) and rec (recall) are sorted in the decreasing order of the score.
                f1 is the max f1 score of those precison-recall points
        """
        sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)
        prec = []
        rec = []
        correct = 0
        total = len(self.facts)
        for i, item in enumerate(sorted_pred_result):
            if (item['entpair'][0], item['entpair'][1], item['relation']) in self.facts:
                correct += 1
            prec.append(float(correct) / float(i + 1))
            rec.append(float(correct) / float(total))
        auc = sklearn.metrics.auc(x=rec, y=prec)
        np_prec = np.array(prec)
        np_rec = np.array(rec)
        f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()
        mean_prec = np_prec.mean()
        result = {'micro_p': np_prec, 'micro_r': np_rec, 'micro_p_mean': mean_prec, 'micro_f1': f1, 'pr_auc': auc}
        return result


def BagRELoader(path, rel2id, tokenizer, batch_size,
                shuffle, entpair_as_bag=False, bag_size=0, num_workers=8,
                collate_fn=BagREDataset.collate_fn):
    if bag_size == 0:
        collate_fn = BagREDataset.collate_fn
    else:
        collate_fn = BagREDataset.collate_bag_size_fn
    dataset = BagREDataset(path, rel2id, tokenizer, entpair_as_bag=entpair_as_bag, bag_size=bag_size)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return data_loader
