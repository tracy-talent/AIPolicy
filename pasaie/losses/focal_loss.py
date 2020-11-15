"""
 Author: liujian
 Date: 2020-11-15 21:47:16
 Last Modified by: liujian
 Last Modified time: 2020-11-15 21:47:16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2., weight=None, reduction='mean', ignore_index=-100):
        """
        Args:
            gamma (float, optional): focus parameter: power of (1-p). Defaults to 2..
            weight (torch.tensor, optional): loss weight. Defaults to None.
            reduction (str, optional): reduction manner. Defaults to 'mean'.
            ignore_index (int, optional): loss will ignore sample whose label is ignore_index. Defaults to -100.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ignore_index=ignore_index


    def forward(self, output, target, mask=None):
        """
        Args:
            output (torch.tensor): (B, C, [d1,d2..])
            target (torch.tensor): (B, [d1,d2..])
            mask (torch.tensor, optional): (B, [d1,d2..]). Defaults to None.
        Returns:
            loss [torch.tensor]: loss, size decideed by reduction manner
        """
        logpt = F.log_softmax(output, dim=1) # log_softmax is more secure than softmax->log
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight, reduction='none', ignore_index=self.ignore_index)

        if self.reduction == 'sum':
            if mask is not None:
                loss = loss * mask
            loss = loss.sum()
        elif self.reduction == 'mean':
            if mask is not None:
                loss = (loss * mask).sum(dim=-1) / mask.sum(dim=-1)
            loss = loss.mean()
        elif self.reduction != 'none':
            raise ValueError("Invalid reduction. Must be 'sum' or 'mean' or 'non'.")
        return loss
