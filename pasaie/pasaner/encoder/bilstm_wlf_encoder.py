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
                blank_padding=True, 
                batch_first=True):
        """
        Args:
            pretrain_path: path of pretrain model
        """
        super(BILSTM_WLFEncoder, self).__init__(char2id, word2id, max_length, hidden_size, char_size, word_size, char2vec, word2vec, custom_dict, blank_padding)
        self.bilstm = nn.LSTM(input_size=self.input_size, 
                            hidden_size=self.hidden_size, 
                            num_layers=1, 
                            bidirectional=True, 
                            batch_first=batch_first)
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
        # Check size of tensors
        inputs_embed = torch.cat([
            self.char_embedding(seqs_char),
            self.word_embedding(seqs_word)
        ], dim=-1) # (B, L, EMBED)
        inputs_length = att_mask.sum(dim=-1)
        inputs_embedding_packed = pack_padded_sequence(inputs_embedding, inputs_length, batch_first=self.batch_first)
        inputs_hiddens_packed, _ = self.bilstm(inputs_embedding_packed)
        inputs_hiddens, _ = pack_padded_sequence(inputs_hiddens_packed, batch_first=self.batch_first)
        # inputs_hiddens = self.bilstm(inputs_embed)
        return inputs_hiddens
    
    def tokenize(self, text):
        return super().tokenize(text)