import torch
import torch.nn as nn
import torch.nn.functional as F

from pasaie.utils.attention import MultiHeadedAttention
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BilstmAttn(nn.Module):

    def __init__(self, sequence_encoder, num_class, embedding_dim, hidden_size=128, num_layers=1, num_heads=8,
                 dropout_rate=0.2, compress_seq=False, batch_first=True, use_attn=True):
        super(BilstmAttn, self).__init__()
        self.sequence_encoder = sequence_encoder
        self.num_classes = num_class
        self.compress_seq = compress_seq
        self.batch_first = batch_first
        self.use_attn = use_attn

        self.bilstm1 = nn.LSTM(input_size=embedding_dim,
                               num_layers=num_layers,
                               hidden_size=hidden_size,
                               dropout=0,
                               bidirectional=True,
                               batch_first=self.batch_first)
        if use_attn:
            self.bilstm2 = nn.LSTM(input_size=hidden_size,
                                   num_layers=num_layers,
                                   hidden_size=hidden_size,
                                   dropout=0,
                                   bidirectional=True,
                                   batch_first=self.batch_first)

            self.attn = MultiHeadedAttention(num_heads=num_heads,
                                             d_model=hidden_size,
                                             dropout=0.1)

        self.fc = nn.Linear(hidden_size * 2, num_class, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, *args):
        if not hasattr(self, '_flattened'):
            self.bilstm1.flatten_parameters()
            if self.use_attn:
                self.bilstm2.flatten_parameters()
            setattr(self, '_flattened', True)
        rep = self.sequence_encoder(*args)
        if self.compress_seq:
            att_mask = args[-1]
            seqs_length = att_mask.sum(dim=-1).detach().cpu()
            seqs_rep_packed = pack_padded_sequence(rep, seqs_length, batch_first=self.batch_first)
            seqs_hiddens_packed, _ = self.bilstm1(seqs_rep_packed)
            seqs_hiddens, _ = pad_packed_sequence(seqs_hiddens_packed, batch_first=self.batch_first)  # B, S, D
        else:
            seqs_length = None
            seqs_hiddens, _ = self.bilstm1(rep)
        x = torch.add(*seqs_hiddens.chunk(2, dim=-1))

        if self.use_attn:
            x = self.attn(x, x, x, mask=args[-1])
            if self.compress_seq:
                seqs_rep_packed2 = pack_padded_sequence(x, seqs_length, batch_first=self.batch_first)
                seqs_hiddens_packed2, _ = self.bilstm2(seqs_rep_packed2)
                x, _ = pad_packed_sequence(seqs_hiddens_packed2, batch_first=self.batch_first)  # B, S, D
            else:
                x, _ = self.bilstm2(x)
            x = torch.add(*x.chunk(2, dim=-1))
        x_last = torch.cat((x[:, 0, :], x[:, -1, :]), dim=-1)  # select the first and the last layer's output
        logits = self.fc(x_last)
        return logits

    def infer(self, item):
        self.eval()
        item = self.sequence_encoder.tokenize(item)
        logits = self.forward(*item)
        if self.num_classes == 2:
            logits = F.sigmoid(logits)
        else:
            logits = F.softmax(logits, dim=-1)
        score, pred = logits.max(-1)
        score = score.item()
        pred = pred.item()
        return pred, score
