"""
 Author: liujian
 Date: 2020-11-26 12:02:15
 Last Modified by: liujian
 Last Modified time: 2020-11-26 12:02:15
"""

import torch
import torch.nn as nn


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        """
        Args:
            num (int, optional): the number of loss. Defaults to 2.
        """
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
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
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum