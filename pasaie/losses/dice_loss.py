"""
 Author: liujian
 Date: 2020-11-15 21:47:13
 Last Modified by: liujian
 Last Modified time: 2020-11-15 21:47:13
"""

import torch
from torch import nn

class DiceLoss(nn.Module):
    """DiceLoss implemented from 'Dice Loss for Data-imbalanced NLP Tasks'
    Useful in dealing with unbalanced data
    """
    def __init__(self, alpha=0.6, gamma=0., reduction='mean'):
        """
        Args:
            alpha (float, optional): focus parameter: power of (1-p), finetune range(0.1~0.9). Defaults to 0.6.
            gamma ([type], optional): smoothing factor, allow to 0. Defaults to 0..
            reduction (str, optional): reduction manner. Defaults to 'mean'.
        """
        super(DiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction


    def forward(self, output, target, mask=None):
        """
        Args:
            output (torch.tensor): (B, C, [d1,d2..])
            target (torch.tensor): (B, [d1,d2..])
            mask (torch.tensor, optional): (B, [d1,d2..]). Defaults to None.
        Returns:
            loss [torch.tensor]: loss, size decideed by reduction manner
        """
        prob = torch.softmax(output, dim=1)
        prob = torch.gather(prob, dim=1, index=target.unsqueeze(1)).squeeze(1)
        prob_with_factor = ((1 - prob) ** self.alpha) * prob
        loss = 1 - (2 * prob_with_factor + self.gamma) / (prob_with_factor + 1 + self.gamma)

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