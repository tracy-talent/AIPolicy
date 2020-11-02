import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .base_encoder import BaseEncoder


class BILSTMEncoder(BaseEncoder):
    def __init__(self, 
                token2id, 
                max_length=256, 
                hidden_size=230, 
                word_size=50, 
                word2vec=None, 
                compress_seq=True,
                blank_padding=True, 
                batch_first=True):
        """
        Args:
            compress_seq (bool, optional): whether compress sequence for lstm. Defaults to True.
        """
        super(BILSTMEncoder, self).__init__(token2id, max_length, hidden_size, word_size, word2vec, blank_padding)
        self.bilstm = nn.LSTM(input_size=self.input_size, 
                            hidden_size=self.hidden_size, 
                            num_layers=1, 
                            bidirectional=True, 
                            batch_first=batch_first)
        self.compress_seq = compress_seq
        self.batch_first = batch_first

    def forward(self, seqs, att_mask):
        """
        Args:
            seqs: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        seqs_embedding = self.word_embedding(seqs)
        if self.compress_seq:
            seqs_length = att_mask.sum(dim=-1).detach().cpu()
            seqs_embedding_packed = pack_padded_sequence(seqs_embedding, seqs_length, batch_first=self.batch_first)
            seqs_hiddens_packed, _ = self.bilstm(seqs_embedding_packed)
            seqs_hiddens, _ = pad_packed_sequence(seqs_hiddens_packed, batch_first=self.batch_first)
        else:
            seqs_hiddens, _ = self.bilstm(seqs_embedding)
        seqs_hiddens = torch.add(*seqs_hiddens.chunk(2, dim=-1))
        return seqs_hiddens
    
    def tokenize(self, *items):
        return super().tokenize(items)