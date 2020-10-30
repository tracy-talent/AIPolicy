import json
import numpy as np
from torch.utils.data import WeightedRandomSampler


def get_sampler(train_path,
                rel2id,
                sampler_type):
    """
        Get a self-defined sampler for train-files.
    :param train_path: str, path of train-files
    :param rel2id: dict, a rel2id dictionary
    :param sampler_type: str, select sampler type
    :return:
    """
    data = []
    labels = []
    with open(train_path, 'r', encoding='utf8') as fin:
        for line in fin.readlines():
            dic = json.loads(line)
            data.append(dic['token'])
            labels.append(rel2id[dic['relation']])

    if sampler_type == "WeightedRandomSampler":
        # Attention: minimum label index must be 0
        label_weight = [1.0 / len(np.where(labels == l)[0])**0.5 for l in np.unique(labels)]
        weights = [label_weight[l] for l in labels]
        return WeightedRandomSampler(weights=weights,
                                     num_samples=len(labels),
                                     replacement=True)
    else:
        raise NotImplementedError('{} has not been implemented'.format(sampler_type))
