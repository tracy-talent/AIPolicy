import torch
import math
import copy
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)
        self.attention_weights = None
        self.dropout = nn.Dropout(p=dropout)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)

        Args:
            x (torch.tensor): tensor to be splited
            batch_size (int): batch size.
        """
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        """

        Args:
            query (torch.tensor): query tensor.
            key (torch.tensor): key tensor.
            value (torch.tensor): value tensor.
            mask (torch.tensor, optional): sequence mask tensor. Defaults to None.

        Returns:
            attention_outputs (torch.tensor): output of MultiHeadedAttention.
        """
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # 2) Apply attention on all the projected vectors in batch.
        attention_outputs, self.attention_weights = scaled_dot_product_attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        attention_outputs = attention_outputs.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        attention_outputs = self.dense(attention_outputs)

        return attention_outputs


def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask[:, None, None, :] # (B, 1, 1, d_k)
        attention_scores.masked_fill_(mask == 0, -1e9)
    attention_weights = F.softmax(attention_scores, dim=-1)
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    attention_outputs = torch.matmul(attention_weights, value)
    return attention_outputs, attention_weights
