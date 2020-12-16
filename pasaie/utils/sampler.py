import json
import numpy as np
from torch.utils.data import WeightedRandomSampler
import pandas as pd


def get_relation_sampler(train_path,
                         rel2id,
                         sampler_type,
                         default_factor=0.5):
    """
        Get a self-defined sampler for train-files.
    :param train_path: str, path of train-files
    :param rel2id: dict, a rel2id dictionary
    :param sampler_type: str, select sampler type
    :return:
    """
    data = []
    labels = []
    with open(train_path, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            dic = json.loads(line)
            data.append(dic['token'])
            labels.append(rel2id[dic['relation']])
    labels = np.array(labels)

    if sampler_type == "WeightedRandomSampler":
        # Attention: minimum label index must be 0
        label_weight = [1.0 / len(np.where(labels == l)[0]) ** default_factor for l in np.unique(labels)]
        weights = [label_weight[l] for l in labels]
        return WeightedRandomSampler(weights=weights,
                                     num_samples=len(labels),
                                     replacement=True)
    else:
        raise NotImplementedError('{} has not been implemented'.format(sampler_type))


def get_entity_span_single_sampler(train_path,
                                   tag2id,
                                   encoder,
                                   max_span,
                                   sampler_type,
                                   default_factor=0.5):
    """
        Get a self-defined sampler for train-files.
    :param train_path: str, path of train-files
    :param tag2id: dict, a tag2id dictionary
    :param encoder: torch.nn.Module, encoder model
    :param max_span: int, max length of entity span
    :param sampler_type: str, select sampler type
    :return:
    """
    labels = []
    skip_cls, skip_sep = 0, 0
    if 'bert' in encoder.__class__.__name__.lower():
        skip_cls, skip_seq = 1, 1
    if 'null' not in tag2id:
        raise Exception(f"negative tag is not null")
    with open(train_path, 'r', encoding='utf-8') as f:
        tokens = []
        for line in f.readlines():
            line = line.strip().split()
            if len(line) > 0:
                tokens.append(line)
            elif len(tokens) > 0:
                items = [list(seq) for seq in zip(*tokens)]
                seqs = list(encoder.tokenize(*items))  # (index_tokens:(1,S), att_maks(1,S))
                seq_len = min(seqs[0].size(1), seqs[-1].sum().item())
                ub = min(max_span, seq_len - skip_cls - skip_sep)
                span_start, span_end = [], []
                for i in range(2, ub + 1):
                    for j in range(skip_cls, seq_len - i + 1 - skip_sep):
                        flag = True
                        if items[1][j][0] == 'B' and items[1][j + i - 1][0] == 'E' and items[1][j][2:] == items[1][
                                                                                                              j + i - 1][
                                                                                                          2:]:
                            for k in range(j + 1, j + i - 1):
                                if items[1][k][0] != 'M':
                                    flag = False
                                    break
                        else:
                            flag = False
                        if flag:
                            labels.append(tag2id[items[1][j][2:]])
                        else:
                            labels.append(tag2id['null'])
                tokens = []
    labels = np.array(labels)

    if sampler_type == "WeightedRandomSampler":
        # Attention: minimum label index must be 0
        label_weight = [1.0 / len(np.where(labels == l)[0]) ** default_factor for l in np.unique(labels)]
        weights = [label_weight[l] for l in labels]
        return WeightedRandomSampler(weights=weights,
                                     num_samples=len(labels),
                                     replacement=True)
    else:
        raise NotImplementedError('{} has not been implemented'.format(sampler_type))


def get_sentence_importance_sampler(train_label,
                                    sampler_type,
                                    default_factor=0.5):
    label = np.array(train_label)
    if sampler_type == "WeightedRandomSampler":
        # Attention: minimum label index must be 0
        label_weight = [1.0 / len(np.where(label == l)[0]) ** default_factor for l in np.unique(label)]
        weights = [label_weight[l] for l in label]
        return WeightedRandomSampler(weights=weights,
                                     num_samples=len(label),
                                     replacement=True)
    else:
        raise NotImplementedError('{} has not been implemented'.format(sampler_type))
