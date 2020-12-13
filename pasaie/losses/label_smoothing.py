"""
 Author: liujian
 Date: 2020-11-15 21:47:10
 Last Modified by: liujian
 Last Modified time: 2020-11-15 21:47:10
"""

import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smooth Regularization
    """
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        """
        Args:
            eps (float, optional): the weight of prior distribution loss. Defaults to 0.1.
            reduction (str, optional): reduction manner. Defaults to 'mean'.
            ignore_index (int, optional): loss will ignore sample whose label is ignore_index. Defaults to -100.
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index


    def forward(self, output, target, mask=None):
        """
        Args:
            output (torch.tensor): (B, C, [d1,d2..])
            target (torch.tensor): (B, [d1,d2..])
            mask (torch.tensor, optional): (B, [d1,d2..]). Defaults to None.
        Returns:
            loss [torch.tensor]: loss, size decideed by reduction manner
        """
        label_num = output.size()[1]
        log_preds = F.log_softmax(output, dim=1)
        prior_loss = -log_preds.sum(dim=1)
        label_loss = F.nll_loss(log_preds, target, reduction='none', ignore_index=self.ignore_index)

        if self.reduction == 'sum':
            if mask is not None:
                prior_loss = prior_loss * mask
                label_loss = label_loss * mask
            prior_loss = prior_loss.sum()
            label_loss = label_loss.sum()
        elif self.reduction == 'mean':
            if mask is not None:
                prior_loss = (prior_loss * mask).sum(dim=-1) / mask.sum(dim=-1)
                label_loss = (label_loss * mask).sum(dim=-1) / mask.sum(dim=-1)
            prior_loss = prior_loss.mean()
            label_loss = label_loss.mean()
        elif self.reduction != 'none':
            raise ValueError("Invalid reduction. Must be 'sum' or 'mean' or 'non'.")

        loss = self.eps * prior_loss / label_num + (1 - self.eps) * label_loss
        return loss