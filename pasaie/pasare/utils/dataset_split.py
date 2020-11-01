#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2020/10/19 17:20
# @Author:  Mecthew
import os
import json
import math
import numpy as np
import sys

sys.path.append('../../..')
from collections import defaultdict
from pasaie.utils import get_logger
import configparser

logger = get_logger(sys.argv)
project_path = '/'.join(os.path.abspath(__file__).split('/')[:-4])
config = configparser.ConfigParser()
config.read(os.path.join(project_path, 'config.ini'))


def train_test_split(corpus_dir,
                     output_name,
                     use_val=True,
                     rel_direction=False,
                     max_samples_num=None,
                     min_sample_num=None,
                     train_ratio=0.6,
                     val_ratio=0.2,
                     test_ratio=0.2
                     ):
    """
        Read corpus and split file into train, valid, test files.
    :param corpus_dir:
    :param output_name: str, num of output directory and its corresponding sub-files
    :param use_val:
    :param rel_direction:
    :param max_samples_num:
    :param min_sample_num:
    :param train_ratio:
    :param val_ratio:
    :param test_ratio:
    :return:
    """
    if use_val and train_ratio + val_ratio + test_ratio != 1:
        raise ValueError("Sum of train, val, test ratios must be 1 when use validation.")
    elif not use_val and train_ratio + test_ratio != 1:
        raise ValueError("Sum of train, test ratios must be 1 when do not use validation.")

    with open(corpus_dir, 'r', encoding='utf8') as fin:
        relation_indices = defaultdict(list)
        rel2id = {}
        samples = []
        total_num = 0
        for idx, line in enumerate(fin.readlines()):
            dic = json.loads(line)
            total_num += 1
            relation_name = dic['relation']
            if rel_direction is False:
                relation_name = relation_name.split('(')[0]
                dic['relation'] = relation_name
            relation_indices[relation_name].append(idx)

            samples.append(dic)
            if relation_name not in rel2id:
                rel2id[relation_name] = len(rel2id.keys())

        train_indices, val_indices, test_indices = [], [], []
        for rel, indices_list in relation_indices.items():
            origin_indices = indices_list.copy()
            if max_samples_num:
                if len(indices_list) > max_samples_num:
                    indices_list = np.random.choice(indices_list, replace=False, size=max_samples_num)
                rel_train = np.random.choice(indices_list, size=int(len(indices_list) * train_ratio), replace=False)
                indices_list = list(set(origin_indices) - set(rel_train))
            else:
                rel_train = np.random.choice(indices_list, size=int(len(indices_list) * train_ratio), replace=False)
                indices_list = list(set(indices_list) - set(rel_train))

            if use_val:
                rel_val = np.random.choice(indices_list,
                                           size=int(len(indices_list) * (val_ratio / (val_ratio + test_ratio))),
                                           replace=False)
                rel_test = list(set(indices_list) - set(rel_val))
            else:
                rel_val = []
                rel_test = list(set(indices_list) - set(rel_train))

            # resample
            if min_sample_num and len(rel_train) < min_sample_num:
                rel_train = list(rel_train)
                rel_train *= math.ceil(min_sample_num / len(rel_train))
            train_indices += list(rel_train)
            val_indices += list(rel_val)
            test_indices += list(rel_test)

        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)
        logger.info("train samples {}; val samples {}; test_samples {}; total_samples {}".format(len(train_indices),
                                                                                                 len(val_indices),
                                                                                                 len(test_indices),
                                                                                                 total_num))
        samples = np.array(samples)
        samples_hash = {'train': train_indices, 'val': val_indices, 'test': test_indices}
        output_dir = os.path.join(config['path']['input'], 'benchmark', 'relation', output_name)
        os.makedirs(output_dir, exist_ok=True)

        for k, v in samples_hash.items():
            corresponding_samples = list(samples[v])

            if len(v) > 0:
                with open(r'{}/{}_{}.txt'.format(output_dir, output_name, k), 'w', encoding='utf8') as fout:
                    # json.dump(corresponding_samples, fout, ensure_ascii=False)
                    for each_sample in corresponding_samples:
                        fout.write(json.dumps(each_sample, ensure_ascii=False) + '\n')

        # write rel2id file
        with open(r'{}/{}_rel2id.json'.format(output_dir, output_name), 'w', encoding='utf8') as fout:
            json.dump(rel2id, fout, indent=1, ensure_ascii=False)


def merge_files(filepath1, filepath2):
    with open(filepath1.split('.')[0] + "-merge-" + filepath2.split('/')[-1].split('\\')[-1], 'w',
              encoding='utf8') as fout:
        f1 = open(filepath1, 'r', encoding='utf8')
        f2 = open(filepath2, 'r', encoding='utf8')
        for line in f1:
            fout.write(line)
        for line in f2:
            fout.write(line)


if __name__ == '__main__':
    train_test_split(r'/home/liujian/qmc_policy/rawdata/re-cleaned/train-merge-val.txt',
                     output_name='test-noval',
                     use_val=True,
                     rel_direction=False,
                     max_samples_num=None,
                     min_sample_num=None,
                     train_ratio=0.6,
                     val_ratio=0.2,
                     test_ratio=0.2
                     )
