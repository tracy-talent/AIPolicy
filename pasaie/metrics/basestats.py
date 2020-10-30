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
