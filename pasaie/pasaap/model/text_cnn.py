import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TextCnn(nn.Module):

    def __init__(self, sequence_encoder, num_class, num_filter, 
                 dropout_rate=0.5, kernel_sizes=None, batch_first=True):
        super(TextCnn, self).__init__()
        self.num_classes = num_class
        self.batch_first = batch_first
        if kernel_sizes is None:
            kernel_sizes = [3, 4, 5]
        self.sequence_encoder = sequence_encoder
        self.convs = nn.ModuleList([nn.Conv1d(sequence_encoder.hidden_size, num_filter, kz) for kz in kernel_sizes])

        self.fc = nn.Linear(len(kernel_sizes) * num_filter, num_class, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, *args):
        x = self.sequence_encoder(*args)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool1d(x_conv, x_conv.size(2)).squeeze(2) for x_conv in x]
        x = torch.cat(x, dim=-1)
        logits = self.fc(x)  # (B, N)
        return logits

    def infer(self, item):
        self.eval()
        item = self.sequence_encoder.tokenize(item)
        logits = self.forward(*item)
        if self.num_classes == 1:
            score = F.sigmoid(logits.squeeze(-1)).item()
            pred = (score >= 0.5)
            score = score if pred else 1 - score
        else:
            logits = F.softmax(logits, dim=-1)
            score, pred = logits.max(-1)
            score = score.item()
            pred = pred.item()
        return pred, score
