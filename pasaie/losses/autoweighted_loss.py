"""
 Author: liujian
 Date: 2020-11-26 12:02:15
 Last Modified by: liujian
 Last Modified time: 2020-11-26 12:02:15
"""

import torch
import torch.nn as nn


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task(classification task) loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2, mode='cls'):
        """
        Args:
            num (int, optional): the number of loss. Defaults to 2.
            mode (str, optional): 'cls' for classification multi-task, 'reg' for regression multi-task. Defaults to 'cls'.
        """
        super(AutomaticWeightedLoss, self).__init__()
        self.mode = mode
        if self.mode not in ['cls', 'reg']:
            raise ValueError('mode argument must be cls or reg.')
        params = torch.ones(num, requires_grad=True) # log(\sigma^2)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        """[summary]

        Args:
            x (tuple or list): multi-task loss
        Returns:
            loss_sum (torch.tensor): sum of multi-task loss
        """
        loss_sum = 0
        for i, loss in enumerate(x):
            if self.mode == 'cls':
                loss_sum += 2 * torch.exp(-self.params[i]) * loss + self.params[i]
            else:
                loss_sum += torch.exp(-self.params[i]) * loss + self.params[i]
        loss_sum /= 2
        return loss_sum
