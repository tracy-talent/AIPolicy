import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .base_wlf_encoder import BaseWLFEncoder

class BILSTM_WLF_Encoder(BaseWLFEncoder):
    def __init__(self, 
                char2id, 
                word2id,
                max_length=256, 
                hidden_size=230, 
                char_size=50,
                word_size=50,
                char2vec=None, 
                word2vec=None, 
                custom_dict=None,
                compress_seq=True,
                blank_padding=True, 
                batch_first=True):
        """
        Args:
            compress_seq (bool, optional): whether compress sequence for lstm. Defaults to True.
        """
        super(BILSTM_WLFEncoder, self).__init__(char2id, word2id, max_length, hidden_size, char_size, word_size, char2vec, word2vec, custom_dict, blank_padding)
        self.bilstm = nn.LSTM(input_size=self.input_size, 
                            hidden_size=self.hidden_size, 
                            num_layers=1, 
                            bidirectional=True, 
                            batch_first=batch_first)
        self.compress_seq = compress_seq
        self.batch_first = batch_first

    def forward(self, seqs_char, seqs_word, att_mask):
        """
        Args:
            seqs_char: (B, L), index of chars
            seqs_word: (B, L), index of words
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, L, H), representations for sequences
        """
        if not hasattr(self, '_flattened'):
            self.bilstm.flatten_parameters()
            setattr(self, '_flattened', True)
        # Check size of tensors
        inputs_embed = torch.cat([
            self.char_embedding(seqs_char),
            self.word_embedding(seqs_word)
        ], dim=-1) # (B, L, EMBED)
        if self.compress_seq:
            inputs_length = att_mask.sum(dim=-1).detach().cpu()
            inputs_embedding_packed = pack_padded_sequence(inputs_embedding, inputs_length, batch_first=self.batch_first)
            inputs_hiddens_packed, _ = self.bilstm(inputs_embedding_packed)
            inputs_hiddens, _ = pad_packed_sequence(inputs_hiddens_packed, batch_first=self.batch_first)
        else:
            inputs_hiddens, _ = self.bilstm(inputs_embed)
        inputs_hiddens = torch.add(*inputs_hiddens.chunk(2, dim=-1))
        return inputs_hiddens
    
    def tokenize(self, *items):
        return super().tokenize(items)