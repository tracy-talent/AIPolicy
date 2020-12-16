from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from functools import partial
import torch
import numpy as np
from collections import Counter

from ...pasare.framework.data_loader import compress_sequence
from ...utils import sampler as mysampler


class SentenceImportanceDataset(data.Dataset):

    def __init__(self, sequence_encoder, data_with_label, is_training, data_augmentation=True):
        self.sequence_encoder = sequence_encoder
        self.is_training = is_training
        self.data_augmentation = data_augmentation
        self.data, self.data_split_indices = self._construct_data(data_with_label)
        labels = np.array(list(zip(*data_with_label))[1])
        self.num_classes = len(np.unique(labels))
        self.weight = np.zeros(self.num_classes, dtype=np.float32)
        class_cnt = Counter(labels)
        for c, cnt in class_cnt.items():
            self.weight[c] += cnt
        self.weight = 1 / self.weight
        self.weight = torch.from_numpy(self.weight)
        if labels.max() == 1 and is_training:
            self.pos_weight = torch.tensor([labels[labels==1].shape[0] / labels[labels==0].shape[0]])
        else:
            self.pos_weight = None

    def _construct_data(self, data_with_label):
        tmp_data = []
        split_indices = []
        split_tokens = [',', '，']
        split_token_ids = [self.sequence_encoder.tokenizer.convert_tokens_to_ids(token) for token in split_tokens]
        for index in range(len(data_with_label)):
            items = data_with_label[index]  # item = (text, label)
            seqs = list(self.sequence_encoder.tokenize(*items))
            seq_len = seqs[-1].sum().item()
            label = items[1]
            tmp_items = [torch.tensor([label])] + seqs
            tmp_data.append(tmp_items)
            tmp_split_indices = [ith for ith, tid in enumerate(seqs[0][0][:seq_len]) if tid.item() in split_token_ids]
            split_indices.append(tmp_split_indices)
        return tmp_data, split_indices

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        if self.is_training and self.data_augmentation and random.random() < 0.3 and self.data_split_indices[index]:
            text_item = item[1]
            mask = item[-1]
            tidx = random.choice(self.data_split_indices[index])
            if random.random() < 0.5:
                t_text = text_item[:, :tidx + 1]
                t_mask = mask[:, :tidx + 1]
            else:
                t_text = text_item[:, :tidx]
                t_mask = mask[:, :tidx]
            if self.sequence_encoder.blank_padding:
                t_text = torch.cat([t_text, torch.zeros((1, self.sequence_encoder.max_length - t_text.size(-1)))], dim=-1).long()
                t_mask = torch.cat([t_mask, torch.zeros((1, self.sequence_encoder.max_length - t_mask.size(-1)))], dim=-1).long()
            # print(t_text.shape, t_mask.shape)
            item = [item[0], t_text, t_mask]
            return item
        else:
            return item


def get_sentence_importance_dataloader(input_data, sequence_encoder, batch_size, shuffle, is_training, sampler=None,
                                       compress_seq=True, num_workers=8):
    if sampler:
        shuffle = False
    dataset = SentenceImportanceDataset(sequence_encoder, input_data, is_training)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             sampler=sampler,
                             num_workers=num_workers,
                             collate_fn=partial(SentenceImportanceDataset.collate_fn, compress_seq))
    return data_loader


def get_train_val_dataloader(csv_path, sequence_encoder, batch_size, sampler=None, test_size=0.3, compress_seq=True):
    csv_data = pd.read_csv(csv_path)
    full_data = csv_data['text'].values
    full_label = csv_data['label'].values

    X_train, X_val, Y_train, Y_val = train_test_split(full_data, full_label,
                                                      random_state=0, test_size=test_size, stratify=full_label)

    train_data = [(x, y) for x, y in zip(X_train, Y_train)]
    val_data = [(x, y) for x, y in zip(X_val, Y_val)]
    if sampler and isinstance(sampler, str):
        sampler = mysampler.get_sentence_importance_sampler(Y_train, sampler_type=sampler, default_factor=0.5)

    train_loader = get_sentence_importance_dataloader(train_data,
                                                      sequence_encoder=sequence_encoder,
                                                      is_training=True,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      sampler=sampler,
                                                      compress_seq=compress_seq)
    val_loader = get_sentence_importance_dataloader(val_data,
                                                    sequence_encoder=sequence_encoder,
                                                    is_training=False,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    sampler=None,
                                                    compress_seq=compress_seq)
    return train_loader, val_loader

