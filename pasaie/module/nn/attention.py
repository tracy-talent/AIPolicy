"""
 Author: liujian
 Date: 2020-12-29 23:45:49
 Last Modified by: liujian
 Last Modified time: 2020-12-29 23:45:49
"""

import torch
import math
import copy
import torch.nn as nn
import torch.nn.functional as F



def dot_product_attention(att_query, att_kv, att_mask):
    att_score = torch.matmul(att_kv, att_query.unsqueeze(-1)).squeeze(-1)
    att_score[att_mask == 0] = 1e-9
    att_weight = F.softmax(att_score, dim=-1)
    att_output = torch.matmul(att_weight.unsqueeze(-2), att_kv).squeeze(-2)
    return att_output, att_weight.data

def dot_product_attention_with_project(att_query, att_kv, att_mask, project_mat):
        att_kv_project = project_mat(att_kv)
        att_score = torch.matmul(att_kv_project, att_query.unsqueeze(-1)).squeeze(-1)
        att_score[att_mask == 0] = 1e-9
        att_weight = F.softmax(att_score, dim=-1)
        att_output = torch.matmul(att_weight.unsqueeze(-2), att_kv).squeeze(-2)
        return att_output, att_weight.data

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_input, num_heads, d_model, d_ffn, dropout_rate=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.wq = nn.Linear(d_input, d_model)
        self.wk = nn.Linear(d_input, d_model)
        self.wv = nn.Linear(d_input, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(p=dropout_rate)
        )
        self.ffn_layernorm = nn.LayerNorm(d_model)
        self.attention_weights = None


    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)

        Args:
            x (torch.tensor): tensor to be splited
            batch_size (int): batch size.
        """
        x = x.contiguous().view(batch_size, -1, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)


    def scaled_dot_product_attention(self, query, key, value, mask=None):
        """"Compute 'Scaled Dot Product Attention' for MultiHeadedAttention"

        Args:
            query (torch.Tensor): attention query, size(B, h, S, D).
            key (torch.Tensor): attention key, size(B, h, S, D).
            value (torch.Tensor): attention value, size(B, h, S, D)
            mask (torch.Tensor, optional): attention mask in transformer encoder. Defaults to None.

        Returns:
            attention_output (torch.Tensor): attention output matrix.
            attention_weight (torch.Tensor): attention weight matrix.
        """
        
        d_k = query.size(-1)
        attention_score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask[:, None, None, :] # (B, 1, 1, d_k)
            attention_score.masked_fill_(mask == 0, -1e9)
        attention_weight = F.softmax(attention_score, dim=-1)
        attention_output = torch.matmul(attention_weight, value)
        return attention_output, attention_weight


    def point_wise_feed_forward_network(self, hidden_states):
        ffn_output = self.ffn(hidden_states)
        ffn_output = self.ffn_layernorm(hidden_states + ffn_output)
        return ffn_output


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
        attention_output, self.attention_weight = self.scaled_dot_product_attention(query, key, value, mask=mask)

        # 3) "Concat" using a view.
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        
        # 4) Feed Forward Network
        output = self.point_wise_feed_forward_network(attention_output)

        return output



class DotProductAttention(nn.Module):
    def __init__(self, hidde_size):
        """
        Args:
            hidde_size (int): dimension of attention_kv.
        """
        super(DotProductAttention, self).__init__()
        self.query = nn.Linear(hidden_size, 1)

    def forward(self, attention_kv):
        """
        Args:
            attention_kv (torch.Tensor): attention key/value matrix, size(B, S, H).

        Returns:
            attention_output (torch.Tensor): attention output matrix.
            attention_weight (torch.Tensor): attention weight matrix.
        """
        attention_score = self.query(attention_kv).squeeze(-1)
        attention_weight = F.softmax(attention_score, dim=-1)
        attention_output = torch.matmul(attention_weight.unsqueeze(1), attention_kv).squeeze(1)
        return attention_output, attention_weight



class MultiplicativeAttention(nn.Module):
    def __init__(self, hidde_size):
        """
        Args:
            hidde_size (int): dimension of attention_kv.
        """
        super(MultiplicativeAttention, self).__init__()
        self.weight_matrix = nn.Linear(hidden_size, hidden_size)
        self.query = nn.Linear(hidden_size, 1)

    def forward(self, attention_kv):
        """
        Args:
            attention_kv (torch.Tensor): attention key/value matrix, size(B, S, H).

        Returns:
            attention_output (torch.Tensor): attention output matrix.
            attention_weight (torch.Tensor): attention weight matrix.
        """
        attention_score = self.query(self.weight_matrix(attention_kv)).squeeze(-1)
        attention_weight = F.softmax(attention_score, dim=-1)
        attention_output = torch.matmul(attention_weight.unsqueeze(1), attention_kv).squeeze(1)
        return attention_output, attention_weight



class AdditiveAttention(nn.Module):
    def __init__(self, kv_hidden_size, query_hidden_size, attention_hidden_size):
        """
        Args:
            kv_hidden_size (int): dimension of attention_kv.
            query_hidden_size (int): dimension of attention_query.
            attention_hidden_size (int): dimension of attention_hidden_state.
        """
        super(AdditiveAttention, self).__init__()
        self.weight_matrix_kv = nn.Linear(kv_hidden_size, attention_hidden_size)
        self.weight_matrix_query = nn.Linear(query_hidden_size, attention_hidden_size)
        self.weight_vector = nn.Linear(attention_hidden_size, 1)


    def forward(self, attention_kv, attention_query):
        """[summary]

        Args:
            attention_kv (torch.Tensor): attention key/value matrix, size(B, S, H).
            attention_query (torch.Tensor): attention query matrix, size(B, H).

        Returns:
            attention_output (torch.Tensor): attention output matrix.
            attention_weight (torch.Tensor): attention weight matrix.
        """
        if len(attention_query.size()) == 3: # for accelerate
            attention_hidden_state = torch.tanh(self.weight_matrix_kv(attention_kv).unsqueeze(1) + 
                                        self.weight_matrix_query(attention_query).unsqueeze(2))
            attention_score = self.weight_vector(attention_hidden_state).squeeze(-1)
            attention_weight = F.softmax(attention_score, dim=-1)
            attention_output = torch.matmul(attention_weight.unsqueeze(2), attention_kv.unsqueeze(1)).squeeze(2)
        else:
            attention_hidden_state = torch.tanh(self.weight_matrix_kv(attention_kv) + 
                                        self.weight_matrix_query(attention_query).unsqueeze(1))
            attention_score = self.weight_vector(attention_hidden_state).squeeze(-1)
            attention_weight = F.softmax(attention_score, dim=-1)
            attention_output = torch.matmul(attention_weight.unsqueeze(1), attention_kv).squeeze(1)
        return attention_output, attention_weight
