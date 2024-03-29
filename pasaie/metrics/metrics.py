"""
 Author: liujian
 Date: 2020-12-13 21:54:51
 Last Modified by: liujian
 Last Modified time: 2020-12-13 21:54:51
"""

import torch
import numpy as np
import time
np.seterr(divide='ignore',invalid='ignore')

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


# class BatchMetric(object):
#     '''Metric computes accuracy/precision/recall/confusion_matrix with batch updates.'''

#     def __init__(self, num_classes, ignore_classes=[]):
#         """
#         Args:
#             num_classes (int): number of classes.
#             ignore_classes (list, optional): classes should be ignored. Defaults to [].
#         """
#         self.num_classes = num_classes
#         self.pos_classes = [i for i in range(num_classes) if i not in ignore_classes]
#         self.y_pred = torch.tensor([])
#         self.y_true = torch.tensor([])
#         self.steps = 0
#         self.step_time = [time.time()]
#         self.tp, self.fp, self.fn, self.tn = None, None, None, None
#         self.acc, self.prec, self.rec, self.f1 = [], [], [], []
#         self._processed = False
#         self.mask = None

#     def __len__(self):
#         return self.y_true.size(0)

#     def update(self, y_pred, y_true, update_score=True):
#         '''Update with batch outputs and labels.
#         Args:
#           y_pred: (tensor) model outputs sized [N,].
#           y_true: (tensor) labels targets sized [N,].
#           update_score: generally set False for val or test set
#         '''
#         self._processed = False
#         self.y_pred = torch.cat([self.y_pred, y_pred.detach().cpu()], dim=0)
#         self.y_true = torch.cat([self.y_true, y_true.detach().cpu()], dim=0)
#         self.steps += 1
#         self.step_time.append(time.time())
#         if update_score:
#             self.tp, self.fp, self.fn, self.tn = self._process(self.y_pred, self.y_true)

#     def _process(self, y_pred, y_true):
#         '''Compute TP, FP, FN, TN.
#         Args:
#           y_pred: (tensor) model outputs sized [N,].
#           y_true: (tensor) labels targets sized [N,].
#         Returns:
#           (tensor): TP, FP, FN, TN, sized [num_classes,].
#         '''
#         self._processed = True
#         tp = torch.empty(self.num_classes, dtype=torch.float)
#         fp = torch.empty(self.num_classes, dtype=torch.float)
#         fn = torch.empty(self.num_classes, dtype=torch.float)
#         tn = torch.empty(self.num_classes, dtype=torch.float)
#         for i in range(self.num_classes):
#             tp[i] = ((y_pred == i) & (y_true == i)).sum()
#             fp[i] = ((y_pred == i) & (y_true != i)).sum()
#             fn[i] = ((y_pred != i) & (y_true == i)).sum()
#             tn[i] = ((y_pred != i) & (y_true != i)).sum()

#         return tp, fp, fn, tn

#     def accuracy(self):
#         '''Accuracy = (TP+TN) / (P+N).
#         Returns:
#           (tensor) accuracy.
#         '''
#         if len(self) == 0:
#             raise ValueError('y_pred or y_true can not be none')
#         acc = (self.y_true == self.y_pred).sum().float() / self.y_true.size(0)
#         self.acc.append(acc)
#         return acc

#     def precision(self, reduction='micro'):
#         '''Precision = TP / (TP+FP).
#         Args:
#           reduction: (str) mean or none.
#         Returns:
#           (tensor) precision.
#           :param ignore_classes:
#         '''
#         if len(self) == 0:
#             raise ValueError('y_pred or y_true can not be none')
#         assert (reduction in ['none', 'macro', 'micro'])
#         if not self._processed:
#             self.tp, self.fp, self.fn, self.tn = self._process(self.y_pred, self.y_true)
#         tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
#         prec = tp / (tp + fp)
#         prec[torch.isnan(prec) | torch.isinf(prec)] = 0
#         if reduction == 'macro':
#             prec = prec[self.pos_classes].mean()
#         elif reduction == 'micro':
#             prec = tp[self.pos_classes].sum() / (tp + fp)[self.pos_classes].sum()
#         self.prec.append(prec)
#         return prec

#     def recall(self, reduction='micro'):
#         '''Recall = TP / P.
#         Args:
#           reduction: (str) mean or none.
#         Returns:
#           (tensor) recall.
#           :param ignore_classes:
#         '''
#         if len(self) == 0:
#             raise ValueError('y_pred or y_true can not be none')
#         assert (reduction in ['none', 'macro', 'micro'])
#         if not self._processed:
#             self.tp, self.fp, self.fn, self.tn = self._process(self.y_pred, self.y_true)
#         tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
#         recall = tp / (tp + fn)
#         recall[torch.isnan(recall) | torch.isinf(recall)] = 0
#         if reduction == 'macro':
#             recall = recall[self.pos_classes].mean()
#         elif reduction == 'micro':
#             recall = tp[self.pos_classes].sum() / (tp + fn)[self.pos_classes].sum()
#         self.rec.append(recall)
#         return recall

#     def f1_score(self, reduction='micro'):
#         if len(self) == 0:
#             raise ValueError('y_pred or y_true can not be none')
#         assert (reduction in ['none', 'macro', 'micro'])
#         # FIXME: In order to accelerate computation, precision and recall must be computed before
#         assert len(self.f1) + 1 == len(self.prec)
#         f1 = 2 * self.prec[-1] * self.rec[-1] / (self.prec[-1] + self.rec[-1])
#         f1[torch.isnan(f1) | torch.isinf(f1)] = 0.
#         self.f1.append(f1)
#         return f1

#     def confusion_matrix(self) -> torch.tensor:
#         matrix = torch.zeros(self.num_classes, self.num_classes)
#         for i in range(self.num_classes):
#             for j in range(self.num_classes):
#                 matrix[j][i] = ((self.y_pred == i) & (self.y_true == j)).sum().item()
#         return matrix

#     def step_time_interval(self):
#         return self.step_time[-1] - self.step_time[-2]

#     def epoch_time_interval(self):
#         return self.step_time[-1] - self.step_time[0]


class BatchMetric(object):
    '''Metric computes accuracy/precision/recall/confusion_matrix with batch updates.'''

    def __init__(self, num_classes, ignore_classes=[]):
        """
        Args:
            num_classes (int): number of classes.
            ignore_classes (list, optional): classes should be ignored. Defaults to [].
        """
        self.num_classes = num_classes
        self.pos_classes = [i for i in range(num_classes) if i not in ignore_classes]
        self.num_acc = 0
        self.num_samples = 0
        self.matrix = np.zeros((num_classes, num_classes))
        self.tp = np.zeros(num_classes) 
        self.fp = np.zeros(num_classes) 
        self.fn = np.zeros(num_classes) 
        self.tn = np.zeros(num_classes)
        self.acc, self.prec, self.rec, self.f1 = [], [], [], []
        self.steps = 0
        self.step_time = [time.time()]

    def __len__(self):
        return self.num_samples

    def update(self, y_pred, y_true):
        '''Update with batch outputs and labels.
        Args:
          y_pred: (tensor) model outputs sized [N,].
          y_true: (tensor) labels targets sized [N,].
        '''
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        self.steps += 1
        self.step_time.append(time.time())
        self._process(y_pred, y_true)

    def _process(self, y_pred, y_true):
        '''Compute TP, FP, FN, TN.
        Args:
          y_pred: (tensor) model outputs sized [N,].
          y_true: (tensor) labels targets sized [N,].
        Returns:
          (tensor): TP, FP, FN, TN, sized [num_classes,].
        '''
        self.num_samples += len(y_pred)
        self.num_acc += (y_pred == y_true).sum()
        for i in range(self.num_classes):
            self.tp[i] += ((y_pred == i) & (y_true == i)).sum()
            self.fp[i] += ((y_pred == i) & (y_true != i)).sum()
            self.fn[i] += ((y_pred != i) & (y_true == i)).sum()
            self.tn[i] += ((y_pred != i) & (y_true != i)).sum()
        # self.confusion_matrix()

    def accuracy(self):
        '''Accuracy = (TP+TN) / (P+N).
        Returns:
          (tensor) accuracy.
        '''
        if len(self) == 0:
            raise ValueError('num_samples can not be 0')
        acc = self.num_acc / self.num_samples
        self.acc.append(acc)
        return acc

    def precision(self, reduction='micro'):
        '''Precision = TP / (TP+FP).
        Args:
          reduction: (str) mean or none.
        Returns:
          (tensor) precision.
          :param ignore_classes:
        '''
        if len(self) == 0:
            raise ValueError('num_samples can not be 0')
        assert (reduction in ['none', 'macro', 'micro'])
        prec = self.tp / (self.tp + self.fp)
        prec[np.isnan(prec) | np.isinf(prec)] = 0.
        if reduction == 'macro':
            prec = prec[self.pos_classes].mean()
        elif reduction == 'micro':
            prec = self.tp[self.pos_classes].sum() / (self.tp + self.fp)[self.pos_classes].sum()
            if np.isnan(prec) or np.isinf(prec):
                prec = 0.
        self.prec.append(prec)
        return prec

    def recall(self, reduction='micro'):
        '''Recall = TP / P.
        Args:
          reduction: (str) mean or none.
        Returns:
          (tensor) recall.
          :param ignore_classes:
        '''
        if len(self) == 0:
            raise ValueError('num_samples can not be 0')
        assert (reduction in ['none', 'macro', 'micro'])
        recall = self.tp / (self.tp + self.fn)
        recall[np.isnan(recall) | np.isinf(recall)] = 0.
        if reduction == 'macro':
            recall = recall[self.pos_classes].mean()
        elif reduction == 'micro':
            recall = self.tp[self.pos_classes].sum() / (self.tp + self.fn)[self.pos_classes].sum()
            if np.isnan(recall) or np.isinf(recall):
                recall = 0.
        self.rec.append(recall)
        return recall

    def f1_score(self, reduction='micro'):
        if len(self) == 0:
            raise ValueError('num_samples can not be 0')
        assert (reduction in ['none', 'macro', 'micro'])
        # FIXME: In order to accelerate computation, precision and recall must be computed before
        assert len(self.f1) + 1 == len(self.prec)
        f1 = 2 * self.prec[-1] * self.rec[-1] / (self.prec[-1] + self.rec[-1])
        if reduction == 'none':
            f1[np.isnan(f1) | np.isinf(f1)] = 0.
        else:
            if np.isnan(f1) or np.isinf(f1):
                f1 = 0.
        self.f1.append(f1)
        return f1

    def confusion_matrix(self, y_pred, y_true):
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                self.matrix[i][j] += ((y_true == i) & (y_pred == j)).sum()

    def step_time_interval(self):
        return self.step_time[-1] - self.step_time[-2]

    def epoch_time_interval(self):
        return self.step_time[-1] - self.step_time[0]
