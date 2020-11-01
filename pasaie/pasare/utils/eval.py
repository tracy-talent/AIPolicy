#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2020/10/22 19:20
# @Author:  Mecthew
import json
from collections import defaultdict


def _sub_eval(y_true, y_pred):
    """
        Get the evaluation of results.
    :param y_true: list, true labels
    :param y_pred: list, prediction labels
    :return: dict, contains precision, recall and f1-score of each class
    """
    y_true_indices = defaultdict(list)
    y_pred_indices = defaultdict(list)
    for idx, y in enumerate(y_true):
        y_true_indices[y].append(idx)
    for idx, y in enumerate(y_pred):
        y_pred_indices[y].append(idx)

    class_eval = defaultdict(dict)
    for cls in y_true_indices.keys():
        cls_true = y_true_indices[cls]
        cls_pred = y_pred_indices[cls]
        precision_collect_sum = sum([int(y_pred[idx] == y_true[idx]) for idx in cls_pred])
        recall_collect_sum = sum([int(y_pred[idx] == y_true[idx]) for idx in cls_true])
        class_eval[cls]['precision'] = precision_collect_sum / (len(cls_pred) + 1e-5)
        class_eval[cls]['recall'] = recall_collect_sum / (len(cls_true) + 1e-5)
        class_eval[cls]['f1'] = 2 * class_eval[cls]['precision'] * class_eval[cls]['recall'] / (
                    class_eval[cls]['precision'] + class_eval[cls]['recall'] + 1e-5)
    return class_eval


def get_eval(true_path, pred_path, rel_direction=False):
    """
        Read prediction and true labels to get the evaluation of results
    :param true_path:
    :param pred_path:
    :param rel_direction:
    :return:
    """
    with open(true_path, 'r', encoding='utf8') as fin:
        y_true = []
        for line in fin.readlines():
            dic = json.loads(line)
            relation_name = dic['relation']
            if rel_direction is False:
                relation_name = relation_name.split('(')[0]
            y_true.append(relation_name)
    with open(pred_path, 'r') as fin:
        y_pred = [line.strip() for line in fin]

    return _sub_eval(y_true, y_pred)


def read_micro_result(filepath):
    """
        Read micro evaluation of results file.
    :param filepath:
    :return: dict, contains precision, recall and f1-score of micro-results
    """
    with open(filepath, 'r') as fin:
        dic = json.load(fin)
    micro_res = {"micro_result": {'precision': dic['micro_p'], 'recall': dic['micro_r'], 'f1': dic['micro_f1']}}
    return micro_res


if __name__ == '__main__':
    pass
