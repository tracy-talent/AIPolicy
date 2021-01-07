"""
 Author: liujian
 Date: 2021-01-07 21:59:02
 Last Modified by: liujian
 Last Modified time: 2021-01-07 21:59:02
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.):
        super(FeedForwardNetwork, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate, inplace=False)
        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_proj = self.dropout(F.relu(self.linear1(x)))
        x_proj = self.linear2(x_proj)
        return x_proj


class PoolerStartLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerStartLogits, self).__init__()
        self.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, p_mask=None):
        x = self.dense(hidden_states)
        return x


class PoolerEndLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerEndLogits, self).__init__()
        self.dense_0 = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.activation = nn.Tanh()
        self.dense_1 = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, start_positions=None, p_mask=None):
        x = self.dense_0(torch.cat([hidden_states, start_positions], dim=-1))
        x = self.LayerNorm(x)
        x = self.activation(x) # activation after normalization to avoid fall in saturation region
        x = self.dense_1(x)
        return x



class Linear3D(nn.Module):
    def __init__(self, input_size, output_size1, output_size2):
        """[summary]

        Args:
            input_size (int): first dimension of linear weight.
            output_size1 (int): second dimension of linear weight.
            output_size2 (int): third dimension of linear weight.
        """
        super(Linear3D, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_size, output_size1, output_size2))
        self.bias = nn.Parameter(torch.Tensor(output_size1, output_size2))
        self.reset_parameters(self.weight, self.bias)
    
    def reset_parameters(self, weight, bias=None):
        """parameters initializtion

        Args:
            weight (torch.Tensor): linear weight.
            bias (torch.Tensor, optional): linear bias. Defaults to None.
        """
        nn.init.xavier_uniform_(weight)
        if bias is not None:
            fan_in = weight.size(0)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)

    def forward(self, input):
        """[summary]

        Args:
            input (torch.Tensor): left matrix of linear projection.

        Returns:
            output (torch.Tensor): output of linear projection.
        """
        output = torch.einsum('...k, kxy -> ...xy', input, self.weight)
        output = torch.add(output, self.bias)
        return output



class LinearSequence(nn.Module):
    def __init__(self, *sizes):
        super(LinearSequence, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(*sizes))
        self.bias = nn.Parameter(torch.Tensor(*(sizes[:1] + sizes[2:])))
        self.reset_parameters(self.weight, self.bias)

    def reset_parameters(self, weight, bias=None):
        """parameters initializtion

        Args:
            weight (torch.Tensor): linear weight.
            bias (torch.Tensor, optional): linear bias. Defaults to None.
        """
        nn.init.xavier_uniform_(weight)
        if bias is not None:
            fan_in = weight.size(1)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)
    
    def forward(self, input):
        """[summary]

        Args:
            input (torch.Tensor): left matrix of linear projection.

        Returns:
            output (torch.Tensor): output of linear projection.
        """
        seq_len = input.size(1)
        output = torch.einsum('bsd, sd... -> bs...', input, self.weight[:seq_len])
        output = torch.add(output, self.bias[:seq_len])
        return output
