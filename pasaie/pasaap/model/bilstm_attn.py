import torch
import torch.nn as nn
import torch.nn.functional as F

from ...module.nn import MultiHeadedAttention
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BilstmAttn(nn.Module):

    def __init__(self, sequence_encoder, num_class, hidden_size=128, num_layers=1,
                 dropout_rate=0.2, compress_seq=False, batch_first=True, use_attn=True):
        super(BilstmAttn, self).__init__()
        self.lstm = nn.LSTM(sequence_encoder.hidden_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=batch_first)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.out = nn.Linear(hidden_size * 2, num_class)
        self.hidden_map = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.query = nn.Linear(hidden_size * 2, 1) if use_attn else None

        self.sequence_encoder = sequence_encoder
        self.num_classes = num_class
        self.hidden_size = hidden_size
        self.compress_seq = compress_seq
        self.use_attn = use_attn
        self.batch_first = batch_first


    def dot_product_attention(self, attention_kv):
        attention_score = self.query(attention_kv).squeeze(-1)
        attention_weight = F.softmax(attention_score, dim=-1)
        attention_output = torch.matmul(attention_weight.unsqueeze(1), attention_kv).squeeze(1)
        return attention_output, attention_weight.data


    def forward(self, *args):
        x = self.sequence_encoder(*args)
        x = self.dropout(x)

        if self.compress_seq:
            att_mask = args[-1]
            seqs_length = att_mask.sum(dim=-1).detach().cpu()
            seqs_rep_packed = pack_padded_sequence(x, seqs_length, batch_first=self.batch_first)
            output, _ = self.lstm(seqs_rep_packed)
            output, _ = pad_packed_sequence(output, batch_first=True)
        else:
            output, _ = self.lstm(x)

        if self.use_attn:
            output, attention = self.dot_product_attention(output)
        else:
            first_hidden = torch.add(*(output[:, 0, :].squeeze().chunk(2, dim=-1)))
            last_hidden = torch.add(*(output[:, -1, :].squeeze().chunk(2, dim=-1)))
            output = torch.cat([first_hidden, last_hidden], dim=-1)
        return self.out(output)

    def infer(self, item):
        self.eval()
        item = self.sequence_encoder.tokenize(item)
        logits = self.forward(*item)
        if self.num_classes == 1:
            score = F.sigmoid(logits.squeeze(-1)).item()
            pred = (score >= 0.5).long().item()
            score = score if pred else 1 - score
        else:
            logits = F.softmax(logits, dim=-1)
            score, pred = logits.max(-1)
            score = score.item()
            pred = pred.item()
        return pred, score
