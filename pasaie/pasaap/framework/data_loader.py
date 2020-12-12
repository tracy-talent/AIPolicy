from torch.utils import data
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import torch


class SentenceImportanceDataset(data.Dataset):

    def __init__(self, sequence_encoder, data_with_label, is_training):
        self.sequence_encoder = sequence_encoder
        self.is_training = is_training
        self.data = self._construct_data(data_with_label)

    def _construct_data(self, data_with_label):
        tmp_data = []
        for index in range(len(data_with_label)):
            items = data_with_label[index]  # item = [[seq_tokens..], [seq_tags..]]
            seqs = list(self.sequence_encoder.tokenize(items))
            label = items[1]
            item = [torch.tensor([label])] + seqs
            tmp_data.append(item)
        return tmp_data

    @classmethod
    def collate_fn(cls, data):
        seqs = list(zip(*data))
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], dim=0)
        return seqs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def get_sentence_importance_dataloader(input_data, sequence_encoder, batch_size, shuffle, is_training, sampler=None, num_workers=8):
    if sampler:
        shuffle = False
    dataset = SentenceImportanceDataset(sequence_encoder, input_data, is_training)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             sampler=sampler,
                             num_workers=num_workers,
                             collate_fn=SentenceImportanceDataset.collate_fn)
    return data_loader


def get_train_val_dataloader(csv_path, sequence_encoder, batch_size, sampler, test_size=0.3):
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
                                                      sampler=sampler)
    val_loader = get_sentence_importance_dataloader(val_data,
                                                    sequence_encoder,
                                                    is_training=False,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    sampler=None)
    return train_loader, val_loader

