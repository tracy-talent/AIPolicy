from torch.utils import data
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from functools import partial
import torch

from ...pasare.framework.data_loader import compress_sequence


class SentenceImportanceDataset(data.Dataset):

    def __init__(self, sequence_encoder, data_with_label, is_training):
        self.sequence_encoder = sequence_encoder
        self.is_training = is_training
        self.data = self._construct_data(data_with_label)

    def _construct_data(self, data_with_label):
        tmp_data = []
        for index in range(len(data_with_label)):
            item = data_with_label[index]  # item = (text, label)
            seqs = list(self.sequence_encoder.tokenize(item))
            label = item[1]
            tmp_item = [torch.tensor([label])] + seqs
            tmp_data.append(tmp_item)
        return tmp_data

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
        return self.data[index]


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


def get_train_val_dataloader(csv_path, sequence_encoder, batch_size, sampler, test_size=0.3, compress_seq=True):
    csv_data = pd.read_csv(csv_path)
    full_data = csv_data['text'].values
    full_label = csv_data['label'].values

    X_train, X_val, Y_train, Y_val = train_test_split(full_data, full_label,
                                                      random_state=0, test_size=test_size, stratify=full_label)
    train_data = [(x, y) for x, y in zip(X_train, Y_train)]
    val_data = [(x, y) for x, y in zip(X_val, Y_val)]
    train_loader = get_sentence_importance_dataloader(train_data,
                                                      sequence_encoder,
                                                      is_training=True,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      sampler=sampler,
                                                      compress_seq=compress_seq)
    val_loader = get_sentence_importance_dataloader(val_data,
                                                    sequence_encoder,
                                                    is_training=False,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    sampler=None,
                                                    compress_seq=compress_seq)
    return train_loader, val_loader

