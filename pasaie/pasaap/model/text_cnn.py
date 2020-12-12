import torch.nn as nn
import torch.nn.functional as F
import torch


class TextCnn(nn.Module):

    def __init__(self, sequence_encoder, num_class, num_filter, embedding_size,
                 dropout_rate=0.5, kernel_sizes=None):
        super(TextCnn, self).__init__()
        self.num_classes = num_class
        if kernel_sizes is None:
            kernel_sizes = [3, 4, 5]
        self.sequence_encoder = sequence_encoder
        self.convs = nn.ModuleList([nn.Conv1d(embedding_size, num_filter, kz) for kz in kernel_sizes])

        self.fc = nn.Linear(len(kernel_sizes) * num_filter, num_class, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, *args):
        x = self.sequence_encoder(*args)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool1d(x_conv, x_conv.size(2)).squeeze(2) for x_conv in x]
        x = torch.cat(x, dim=-1)

        logits = self.fc(x) # (B, N)
        return logits

