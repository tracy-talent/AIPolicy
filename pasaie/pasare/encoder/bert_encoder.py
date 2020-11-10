import logging
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from .base_encoder import BaseEncoder
import torch.nn.functional as F


class BERTEncoder(nn.Module):
    def __init__(self, max_length, pretrain_path, blank_padding=True, mask_entity=False):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model
        """
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.mask_entity = mask_entity
        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.hidden_size = self.bert.config.hidden_size

    def forward(self, token, att_mask):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        _, x = self.bert(token, attention_mask=att_mask)
        return x

    def tokenize(self, item):
        """
        Args:
            item: data instance containing 'text' / 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        # Sentence -> token
        if 'text' in item:
            sentence = item['text']
            is_token = False
        else:
            sentence = item['token']
            is_token = True
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        pos_min = pos_head
        pos_max = pos_tail
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            rev = False
        
        if not is_token:
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
        else:
            sent0 = self.tokenizer.tokenize(' '.join(sentence[:pos_min[0]]))
            ent0 = self.tokenizer.tokenize(' '.join(sentence[pos_min[0]:pos_min[1]]))
            sent1 = self.tokenizer.tokenize(' '.join(sentence[pos_min[1]:pos_max[0]]))
            ent1 = self.tokenizer.tokenize(' '.join(sentence[pos_max[0]:pos_max[1]]))
            sent2 = self.tokenizer.tokenize(' '.join(sentence[pos_max[1]:]))

        if self.mask_entity:
            ent0 = ['[unused5]'] if not rev else ['[unused6]']
            ent1 = ['[unused6]'] if not rev else ['[unused5]']
        else:
            ent0 = ['[unused1]'] + ent0 + ['[unused2]'] if not rev else ['[unused3]'] + ent0 + ['[unused4]']
            ent1 = ['[unused3]'] + ent1 + ['[unused4]'] if not rev else ['[unused1]'] + ent1 + ['[unused2]']

        re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]']
        pos1 = 1 + len(sent0) if not rev else 1 + len(sent0 + ent0 + sent1)
        pos2 = 1 + len(sent0 + ent0 + sent1) if not rev else 1 + len(sent0)
        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)
        
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        # Position
        pos1 = torch.tensor([[pos1]]).long()
        pos2 = torch.tensor([[pos2]]).long()

        # Padding
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask


class BERTEntityEncoder(nn.Module):
    def __init__(self, max_length, pretrain_path, blank_padding=True, mask_entity=False):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model
        """
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.mask_entity = mask_entity
        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        bert_hidden_size = self.bert.config.hidden_size
        self.hidden_size = bert_hidden_size * 3
        self.conv = nn.Conv2d(1, bert_hidden_size, kernel_size=(5, bert_hidden_size))  # add a convolution layer to extract the global information of sentence
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, token, pos1, pos2, att_mask):
        """
        Args:
            token: (B, L), index of tokens
            pos1: (B, 1), position of the head entity starter
            pos2: (B, 1), position of the tail entity starter
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Returns:
            (B, 2H), representations for sentences
        """
        hidden, _ = self.bert(token, attention_mask=att_mask)
        # Get entity start hidden state
        onehot_head = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_tail = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_head = onehot_head.scatter_(1, pos1, 1)
        onehot_tail = onehot_tail.scatter_(1, pos2, 1)
        head_hidden = (onehot_head.unsqueeze(2) * hidden).sum(1)  # (B, H)
        tail_hidden = (onehot_tail.unsqueeze(2) * hidden).sum(1)  # (B, H)
        # x = torch.cat([head_hidden, tail_hidden], 1)  # (B, 2H)
        context_conv = F.relu(self.conv(hidden.unsqueeze(1))).squeeze(3)
        context_hidden = F.max_pool1d(context_conv, context_conv.size(2)).squeeze(2)
        x = torch.cat([head_hidden, tail_hidden, context_hidden], 1)  # (B, 3H)
        x = self.linear(x)
        return x

    def tokenize(self, item):
        """
        Args:
            item: data instance containing 'text' / 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        # Sentence -> token
        if 'text' in item:
            sentence = item['text']
            is_token = False
        else:
            sentence = item['token']
            is_token = True
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        pos_min = pos_head
        pos_max = pos_tail
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            rev = False
        
        if not is_token:
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
        else:
            sent0 = self.tokenizer.tokenize(' '.join(sentence[:pos_min[0]]))
            ent0 = self.tokenizer.tokenize(' '.join(sentence[pos_min[0]:pos_min[1]]))
            sent1 = self.tokenizer.tokenize(' '.join(sentence[pos_min[1]:pos_max[0]]))
            ent1 = self.tokenizer.tokenize(' '.join(sentence[pos_max[0]:pos_max[1]]))
            sent2 = self.tokenizer.tokenize(' '.join(sentence[pos_max[1]:]))

        if self.mask_entity:
            ent0 = ['[unused5]'] if not rev else ['[unused6]']
            ent1 = ['[unused6]'] if not rev else ['[unused5]']
        else:
            ent0 = ['[unused1]'] + ent0 + ['[unused2]'] if not rev else ['[unused3]'] + ent0 + ['[unused4]']
            ent1 = ['[unused3]'] + ent1 + ['[unused4]'] if not rev else ['[unused1]'] + ent1 + ['[unused2]']

        re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]']
        pos1 = 1 + len(sent0) if not rev else 1 + len(sent0 + ent0 + sent1)
        pos2 = 1 + len(sent0 + ent0 + sent1) if not rev else 1 + len(sent0)
        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)
        
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens) # 序列实际长度

        # Position
        pos1 = torch.tensor([[pos1]]).long()
        pos2 = torch.tensor([[pos2]]).long()

        # Padding
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        return indexed_tokens, pos1, pos2, att_mask
