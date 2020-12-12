import math, logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ...tokenization import WordTokenizer


class BaseEncoder(nn.Module):

    def __init__(self,
                 token2id,
                 max_length=256,
                 embedding_dim=50,
                 word2vec=None,
                 blank_padding=True):
        """
        Args:
            token2id: dictionary of token->idx mapping
            max_length: max length of sentence, used for postion embedding
            embedding_dim: dimension of word embedding
            blank_padding: padding for CNN
            word2vec: pretrained word2vec numpy
        """
        # Hyperparameters
        super().__init__()

        self.token2id = token2id
        self.max_length = max_length
        self.num_token = len(token2id)

        if word2vec is None:
            self.embedding_dim = embedding_dim
        else:
            self.embedding_dim = word2vec.shape[-1]

        self.input_size = self.embedding_dim
        self.blank_padding = blank_padding

        if not '[UNK]' in self.token2id:
            self.token2id['[UNK]'] = len(self.token2id)
            self.num_token += 1
        if not '[PAD]' in self.token2id:
            self.token2id['[PAD]'] = len(self.token2id)
            self.num_token += 1

        # Word embedding
        self.word_embedding = nn.Embedding(self.num_token, self.embedding_dim)
        if word2vec is not None:
            logging.info("Initializing word embedding with word2vec.")
            word2vec = torch.from_numpy(word2vec)
            if self.num_token == len(word2vec) + 2:
                unk = torch.randn(1, self.embedding_dim) / math.sqrt(self.embedding_dim)
                blk = torch.zeros(1, self.embedding_dim)
                self.word_embedding.weight.data.copy_(torch.cat([word2vec, unk, blk], 0))
            else:
                self.word_embedding.weight.data.copy_(word2vec)

        # Position Embedding
        self.tokenizer = WordTokenizer(vocab=self.token2id, unk_token="[UNK]")

    def forward(self, seqs, att_mask):
        """
        Args:
            seqs: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, L, H), representations for sequences
        """
        # Check size of tensors
        inputs_embed = self.word_embedding(seqs)  # (B, L, EMBED)
        return inputs_embed

    def tokenize(self, items):
        """
        Args:
            items: (tokens, tags) or (tokens, spans, atrrs) or (sentence)
        Return:
            index number of tokens and positions             
        """
        if isinstance(items, list) or isinstance(items, tuple):
            sentence = items[0]
        else:
            sentence = items

        # Sentence -> token
        tokens = self.tokenizer.tokenize(sentence)
        avail_len = torch.tensor([len(tokens)])  # 序列实际长度

        # Token -> index
        if self.blank_padding:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens, self.max_length, self.token2id['[PAD]'],
                                                                  self.token2id['[UNK]'])
        else:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens, unk_id=self.token2id['[UNK]'])
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # attention mask
        att_mask = torch.zeros(indexed_tokens.size(), dtype=torch.uint8)  # (1, L), crf reqiure mask dtype=bool or uint8
        att_mask[0, :avail_len] = 1

        return indexed_tokens, att_mask
