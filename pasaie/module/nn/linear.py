import torch
import torch.nn as nn
import torch.nn.functional as F

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
