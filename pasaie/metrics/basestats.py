import torch
import time


class Mean(object):
    """
    Computes and stores the average and current value of metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0 # 最近一个batch的值
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val / n if n > 0 else 0.
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.

    def __str__(self):
        """
        String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class BatchMetric:
    '''Metric computes accuracy/precision/recall/confusion_matrix with batch updates.'''

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.y_pred = []
        self.y_true = []
        self.steps = 0
        self.step_time = [time.time()]

    def __len__(self):
        return torch.cat(self.y_true, 0).size(0)

    def update(self, y_pred, y_true):
        '''Update with batch outputs and labels.
        Args:
          y_pred: (tensor) model outputs sized [N,].
          y_true: (tensor) labels targets sized [N,].
        '''
        self.y_pred.append(y_pred)
        self.y_true.append(y_true)
        self.steps += 1
        self.step_time.append(time.time())

    def _process(self, y_pred, y_true):
        '''Compute TP, FP, FN, TN.
        Args:
          y_pred: (tensor) model outputs sized [N,].
          y_true: (tensor) labels targets sized [N,].
        Returns:
          (tensor): TP, FP, FN, TN, sized [num_classes,].
        '''
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
        if not self.y_pred or not self.y_true:
            return
        y_pred = torch.cat(self.y_pred, 0)
        y_true = torch.cat(self.y_true, 0)
        acc = (y_true == y_pred).sum().float() / y_pred.size(0)
        return acc

    def precision(self, reduction='micro'):
        '''Precision = TP / (TP+FP).
        Args:
          reduction: (str) mean or none.
        Returns:
          (tensor) precision.
        '''
        if not self.y_pred or not self.y_true:
            return
        assert (reduction in ['none', 'macro', 'micro'])
        y_pred = torch.cat(self.y_pred, 0)
        y_true = torch.cat(self.y_true, 0)
        tp, fp, fn, tn = self._process(y_pred, y_true)
        prec = tp / (tp + fp)
        prec[torch.isnan(prec)] = 0
        if reduction == 'macro':
            prec = prec.mean()
        elif reduction == 'micro':
            prec = tp.sum() / (tp + fp).sum()

        return prec

    def recall(self, reduction='micro'):
        '''Recall = TP / P.
        Args:
          reduction: (str) mean or none.
        Returns:
          (tensor) recall.
        '''
        if not self.y_pred or not self.y_true:
            return
        assert (reduction in ['none', 'macro', 'micro'])
        y_pred = torch.cat(self.y_pred, 0)
        y_true = torch.cat(self.y_true, 0)
        tp, fp, fn, tn = self._process(y_pred, y_true)
        recall = tp / (tp + fn)
        recall[torch.isnan(recall)] = 0
        if reduction == 'macro':
            recall = recall.mean()
        elif reduction == 'micro':
            recall = tp.sum() / (tp + fn).sum()
        return recall

    def f1_score(self, reduction='micro'):
        if not self.y_pred or not self.y_true:
            return
        assert (reduction in ['none', 'macro', 'micro'])
        _precision = self.precision(reduction)
        _recall = self.recall(reduction)
        f1 = 2 * _precision * _recall / (_precision + _recall)
        f1[torch.isnan(f1)] = 0.
        return f1

    def confusion_matrix(self) -> torch.tensor:
        y_pred = torch.cat(self.y_pred, 0)
        y_true = torch.cat(self.y_true, 0)
        matrix = torch.zeros(self.num_classes, self.num_classes)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                matrix[j][i] = ((y_pred == i) & (y_true == j)).sum().item()
        return matrix

    def step_time_interval(self):
        return self.step_time[-1] - self.step_time[-2]

    def epoch_time_interval(self):
        return self.step_time[-1] - self.step_time[0]
