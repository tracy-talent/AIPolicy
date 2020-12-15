"""
 Author: liujian
 Date: 2020-12-13 21:54:51
 Last Modified by: liujian
 Last Modified time: 2020-12-13 21:54:51
"""

import torch
import time


def micro_p_r_f1_score(preds, golds):
    """calculate precision/recall/f1 score

    Args:
        preds (list[list]): seq tags predicted by model
        golds (list[list]): gold tags of corpus

    Returns:
        p (float): precision
        r (float): recall
        f1 (float): f1 score
    """
    assert len(preds) == len(golds)
    p_sum = 0
    r_sum = 0
    hits = 0
    for pred, gold in zip(preds, golds):
        p_sum += len(pred)
        r_sum += len(gold)
        for label in pred:
            if label in gold:
                hits += 1
    micro_p = hits / p_sum if p_sum > 0 else 0
    micro_r = hits / r_sum if r_sum > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0
    return micro_p, micro_r, micro_f1


class BatchMetric(object):
    '''Metric computes accuracy/precision/recall/confusion_matrix with batch updates.'''

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.y_pred = torch.tensor([])
        self.y_true = torch.tensor([])
        self.steps = 0
        self.step_time = [time.time()]
        self.tp, self.fp, self.fn, self.tn = None, None, None, None
        self.acc, self.prec, self.rec, self.f1 = [], [], [], []
        self._processed = False

    def __len__(self):
        return self.y_true.size(0)

    def update(self, y_pred, y_true, update_score=True):
        '''Update with batch outputs and labels.
        Args:
          y_pred: (tensor) model outputs sized [N,].
          y_true: (tensor) labels targets sized [N,].
          update_score: generally set False for val or test set
        '''
        self._processed = False
        self.y_pred = torch.cat([self.y_pred, y_pred.detach().cpu()], dim=0)
        self.y_true = torch.cat([self.y_true, y_true.detach().cpu()], dim=0)
        self.steps += 1
        self.step_time.append(time.time())
        if update_score:
            self.tp, self.fp, self.fn, self.tn = self._process(self.y_pred, self.y_true)

    def _process(self, y_pred, y_true):
        '''Compute TP, FP, FN, TN.
        Args:
          y_pred: (tensor) model outputs sized [N,].
          y_true: (tensor) labels targets sized [N,].
        Returns:
          (tensor): TP, FP, FN, TN, sized [num_classes,].
        '''
        self._processed = True
        tp = torch.empty(self.num_classes, dtype=torch.float)
        fp = torch.empty(self.num_classes, dtype=torch.float)
        fn = torch.empty(self.num_classes, dtype=torch.float)
        tn = torch.empty(self.num_classes, dtype=torch.float)
        for i in range(self.num_classes):
            tp[i] = ((y_pred == i) & (y_true == i)).sum().item()
            fp[i] = ((y_pred == i) & (y_true != i)).sum().item()
            fn[i] = ((y_pred != i) & (y_true == i)).sum().item()
            tn[i] = ((y_pred != i) & (y_true != i)).sum().item()

        return tp, fp, fn, tn

    def accuracy(self):
        '''Accuracy = (TP+TN) / (P+N).
        Returns:
          (tensor) accuracy.
        '''
        if len(self) == 0:
            raise ValueError('y_pred or y_true can not be none')

        acc = (self.y_true == self.y_pred).sum().float() / self.y_pred.size(0)
        self.acc.append(acc)
        return acc

    def precision(self, reduction='micro'):
        '''Precision = TP / (TP+FP).
        Args:
          reduction: (str) mean or none.
        Returns:
          (tensor) precision.
        '''
        if len(self) == 0:
            raise ValueError('y_pred or y_true can not be none')
        assert (reduction in ['none', 'macro', 'micro'])
        if not self._processed:
            self.tp, self.fp, self.fn, self.tn = self._process(self.y_pred, self.y_true)
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        prec = tp / (tp + fp)
        prec[torch.isnan(prec)] = 0
        if reduction == 'macro':
            prec = prec.mean()
        elif reduction == 'micro':
            prec = tp.sum() / (tp + fp).sum()
        self.prec.append(prec)
        return prec

    def recall(self, reduction='micro'):
        '''Recall = TP / P.
        Args:
          reduction: (str) mean or none.
        Returns:
          (tensor) recall.
        '''
        if len(self) == 0:
            raise ValueError('y_pred or y_true can not be none')
        assert (reduction in ['none', 'macro', 'micro'])
        if not self._processed:
            self.tp, self.fp, self.fn, self.tn = self._process(self.y_pred, self.y_true)
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        recall = tp / (tp + fn)
        recall[torch.isnan(recall)] = 0
        if reduction == 'macro':
            recall = recall.mean()
        elif reduction == 'micro':
            recall = tp.sum() / (tp + fn).sum()
        self.rec.append(recall)
        return recall

    def f1_score(self, reduction='micro'):
        if len(self) == 0:
            raise ValueError('y_pred or y_true can not be none')
        assert (reduction in ['none', 'macro', 'micro'])
        # FIXME: In order to accelerate computation, precision and recall must be computed before
        assert len(self.f1) + 1 == len(self.prec)
        f1 = 2 * self.prec[-1] * self.rec[-1] / (self.prec[-1] + self.rec[-1])
        f1[torch.isnan(f1)] = 0.
        self.f1.append(f1)
        return f1

    def confusion_matrix(self) -> torch.tensor:
        matrix = torch.zeros(self.num_classes, self.num_classes)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                matrix[j][i] = ((self.y_pred == i) & (self.y_true == j)).sum().item()
        return matrix

    def step_time_interval(self):
        return self.step_time[-1] - self.step_time[-2]

    def epoch_time_interval(self):
        return self.step_time[-1] - self.step_time[0]
