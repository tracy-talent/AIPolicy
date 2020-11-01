#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2020/10/19 16:02
# @Author:  Mecthew

import json
import numpy as np
import sys
import os

sys.path.append('../../..')
from collections import defaultdict
from pasaie.utils import get_logger
import configparser

logger = get_logger(sys.argv)
project_path = '/'.join(os.path.abspath(__file__).split('/')[:-4])
config = configparser.ConfigParser()
config.read(os.path.join(project_path, 'config.ini'))


def relation_statistics(corpus_path):
    """
    :param corpus_path: str, filepath of relation corpus
    :return: multi-tuple, ratios of each relation, average length, max length and min length of sentences
    """
    with open(corpus_path, 'r', encoding='utf8') as fin:
        relation_dict = defaultdict(int)
        total_num = 0
        line_length = []
        for line in fin.readlines():
            dic = json.loads(line)
            relation_name = dic['relation']
            relation_dict[relation_name] += 1
            total_num += 1
            line_length.append(len(dic['token']))

    relation_ratios = sorted([str((k, v, v / total_num)) for k, v in relation_dict.items()], key=lambda x: x[1])
    ave_len = np.mean(line_length)
    max_len = np.max(line_length)
    min_len = np.min(line_length)

    ret_dict = {
        'sent_ave_len': float(ave_len),
        'sent_max_len': float(max_len),
        'sent_min_len': float(min_len),
        'rel_statistic': relation_ratios
    }
    os.makedirs(os.path.join(config['path']['output'], 'relation', 'statistic'), exist_ok=True)
    fout = open(os.path.join(config['path']['output'], 'relation', 'statistic', 're_statistic.json'), 'w', encoding='utf8')
    json.dump(ret_dict, fout, ensure_ascii=False, indent=1)
    return relation_ratios, ave_len, max_len, min_len


def relation_entity_pair_statistic(corpus_path):
    re_entity_pair_dict = defaultdict(dict)
    with open(corpus_path, 'r', encoding='utf8') as fin:
        for line in fin:
            dic = json.loads(line)
            relation_name = dic['relation']
            if relation_name in ['Other', 'other']:
                continue
            head, tail = dic['h'], dic['t']
            if 'head' not in re_entity_pair_dict[relation_name]:
                re_entity_pair_dict[relation_name]['head'] = defaultdict(int)
            if 'tail' not in re_entity_pair_dict[relation_name]:
                re_entity_pair_dict[relation_name]['tail'] = defaultdict(int)
            re_entity_pair_dict[relation_name]['head'][head['entity'] + '-' + head['name']] += 1
            re_entity_pair_dict[relation_name]['tail'][tail['entity'] + '-' + tail['name']] += 1

    ret_dict = defaultdict(dict)
    for relation_name in re_entity_pair_dict.keys():
        head = re_entity_pair_dict[relation_name]['head']
        tail = re_entity_pair_dict[relation_name]['tail']

        head_dict = defaultdict(dict)
        for key, value in head.items():
            entity_type, name = key.split('-')[0], key.split('-')[1]
            if name not in head_dict[entity_type]:
                head_dict[entity_type][name] = value
            else:
                head_dict[entity_type][name] += value
        new_head_dict = {}
        for entity_type in head_dict:
            head_sorted_list = sorted(head_dict[entity_type].items(), key=lambda x: x[1], reverse=True)
            head_sorted_list = [str(item) for item in head_sorted_list]
            new_head_dict[entity_type] = head_sorted_list
            
        tail_dict = defaultdict(dict)
        for key, value in tail.items():
            entity_type, name = key.split('-')[0], key.split('-')[1]
            if name not in tail_dict[entity_type]:
                tail_dict[entity_type][name] = value
            else:
                tail_dict[entity_type][name] += value
        new_tail_dict = {}
        for entity_type in tail_dict:
            tail_sorted_list = sorted(tail_dict[entity_type].items(), key=lambda x: x[1], reverse=True)
            tail_sorted_list = [str(item) for item in tail_sorted_list]
            new_tail_dict[entity_type] = tail_sorted_list

        ret_dict[relation_name]['head'] = new_head_dict
        ret_dict[relation_name]['tail'] = new_tail_dict

    os.makedirs(os.path.join(config['path']['output'], 'relation', 'statistic'), exist_ok=True)
    fout = open(os.path.join(config['path']['output'], 'relation', 'statistic', 're_entity_pairs.json'), 'w', encoding='utf8')
    json.dump(ret_dict, fout, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    # merge_files('data/jiangbei1_v2/jiangbei1_train.txt', 'data/jiangbei1_v2/jiangbei1_val.txt')
    # print(corpus_statistics(r'C:\NLP-Github\PolicyMining\RE\data\jiangbei1_v2\jiangbei1_train-merge-jiangbei1_val.txt'))
    relation_entity_pair_statistic(r'/home/liujian/qmc_policy/rawdata/re-cleaned/train-merge-val.txt')
    relation_statistics(r'/home/liujian/qmc_policy/rawdata/re-cleaned/train-merge-val.txt')
