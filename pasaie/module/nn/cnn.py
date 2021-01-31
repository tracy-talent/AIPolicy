"""
 Author: liujian
 Date: 2021-01-25 10:26:32
 Last Modified by: liujian
 Last Modified time: 2021-01-25 10:26:32
"""

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_singlekernel_conv1d(hiddens, weights, conv):
    """single kernel 1D convolution

    Args:
        hiddens (torch.Tensor): Tensor to be convolved.
        weights (torch.Tensor): mask weights.
        conv (torch.nn.Module): 1D convolution kernel.

    Returns:
        conv_hiddens (torch.Tensor): Tensor after convolution.
    """
    shape = hiddens.size()
    dim1 = functools.reduce(lambda x, y: x * y, shape[:-2])
    dim2 = shape[-2]
    dim3 = shape[-1]
    hiddens = hiddens.contiguous().view(dim1, dim2, dim3).transpose(-2, -1)
    weights = weights.contiguous().view(dim1, dim2, 1).float().transpose(-2, -1)
    hiddens *= weights
    conv_hiddens = conv(hiddens)
    conv_hiddens *= weights
    conv_hiddens = F.relu(F.max_pool1d(conv_hiddens, conv_hiddens.size(-1)).squeeze(-1))
    conv_hiddens = conv_hiddens.contiguous().view(*(shape[:-2] + conv_hiddens.size()[-1:]))
    return conv_hiddens


def masked_multikernel_conv1d(hiddens, weights, convs):
    """mutliple kernel 1D convolution as TextCNN

    Args:
        hiddens (torch.Tensor): Tensor to be convolved.
        weights (torch.Tensor): mask weights.
        convs (torch.nn.ModuleList): multiple 1D convolution kernels.

    Returns:
        conv_hiddens (torch.Tensor): Tensor after convolution.
    """
    shape = hiddens.size()
    dim1 = functools.reduce(lambda x, y: x * y, shape[:-2])
    dim2 = shape[-2]
    dim3 = shape[-1]
    hiddens = hiddens.contiguous().view(dim1, dim2, dim3).transpose(-2, -1)
    weights = weights.contiguous().view(dim1, dim2, 1).float().transpose(-2, -1)
    hiddens *= weights
    convs_out = [conv(hiddens).squeeze(-1) for conv in convs]
    conv_hiddens = torch.cat(convs_out, dim=-1)
    conv_hiddens = conv_hiddens.contiguous().view(*(shape[:-2] + conv_hiddens.size()[-1:]))
    return conv_hiddens


class CNN(nn.Module):

    def __init__(self, input_size=50, hidden_size=256, dropout=0, kernel_size=3, padding=1, activation_function=F.relu):
        """
        Args:
            input_size: dimention of input embedding
            kernel_size: kernel_size for CNN
            padding: padding for CNN
            hidden_size: hidden size
        """
        super().__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size, padding=padding)
        self.act = activation_function
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            input features: (B, L, I_EMBED)
        Return:
            output features: (B, H_EMBED)
        """
        # Check size of tensors
        x = x.transpose(1, 2) # (B, I_EMBED, L)
        x = self.conv(x) # (B, H_EMBED, L)
        x = self.act(x) # (B, H_EMBED, L)
        x = self.dropout(x) # (B, H_EMBED, L)
        x = x.transpose(1, 2) # (B, L, H_EMBED)
        return x
