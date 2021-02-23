"""
 Author: liujian 
 Date: 2021-02-22 09:45:25 
 Last Modified by: liujian 
 Last Modified time: 2021-02-22 09:45:25 
"""

from .base_encoder import BaseEncoder
from ...utils.dependency_parse import DDP_Parse, LTP_Parse, Stanza_Parse

import logging
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import BertModel, RobertaModel, AlbertModel
from transformers import BertTokenizer, RobertaTokenizer, AlbertTokenizer


class BERTEncoder(nn.Module):
    def __init__(self, pretrain_path, bert_name='bert', max_length=256, blank_padding=True, mask_entity=False, language='en'):
        """
        Args:
            pretrain_path (str): path of pretrain model
            bert_name (str, optional): name of pretrained 'bert series' model. Defaults to 'bert'.
            max_length (int, optional): max_length of sequence. Defaults to 256.
            blank_padding (bool, optional): whether pad sequence to the same length. Defaults to True.
            mask_entity (bool, optional): whether do mask for entity. Defaults to False.
        
        Raises:
            NotImplementedError: bert pretrained model is not implemented.
        """
        super().__init__()
        self.language = language
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.mask_entity = mask_entity
        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert_name = bert_name
        if 'albert' in bert_name:
            if self.language == 'zh':
                self.bert = AlbertModel.from_pretrained(pretrain_path) # clue
                self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
            else:
                self.bert = AlbertModel.from_pretrained(pretrain_path)
                self.tokenizer = AlbertTokenizer.from_pretrained(pretrain_path)
        elif 'roberta' in bert_name:
            if self.language == 'zh':
                self.bert = BertModel.from_pretrained(pretrain_path) # clue, hfl
                self.tokenizer = BertTokenizer.from_pretrained(pretrain_path) # clue, hfl
            else:
                self.bert = RobertaModel.from_pretrained(pretrain_path)
                self.tokenizer = RobertaTokenizer.from_pretrained(pretrain_path)
        elif 'bert' in bert_name:
            self.bert = BertModel.from_pretrained(pretrain_path)
            self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        else:
            raise NotImplementedError(f'{bert_name} pretrained model is not implemented!')
        # add possible missed tokens in vocab.txt
        num_added_tokens = self.tokenizer.add_tokens(['“', '”', '—'])
        logging.info(f"we have added {num_added_tokens} tokens ['“', '”', '—']")
        self.bert.resize_token_embeddings(len(self.tokenizer))

        self.hidden_size = self.bert.config.hidden_size

    def forward(self, seqs, att_mask):
        """
        Args:
            seqs: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        if 'roberta' in self.bert_name:
            _, pooler_out = self.bert(seqs, attention_mask=att_mask) # clue-roberta
        else:
            _, pooler_out = self.bert(seqs, attention_mask=att_mask)
        return pooler_out

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
        self.sentence = sentence
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

        join_token = ' ' if self.language == 'en' else ''
        if not is_token:
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
        else:
            sent0 = self.tokenizer.tokenize(join_token.join(sentence[:pos_min[0]]))
            ent0 = self.tokenizer.tokenize(join_token.join(sentence[pos_min[0]:pos_min[1]]))
            sent1 = self.tokenizer.tokenize(join_token.join(sentence[pos_min[1]:pos_max[0]]))
            ent1 = self.tokenizer.tokenize(join_token.join(sentence[pos_max[0]:pos_max[1]]))
            sent2 = self.tokenizer.tokenize(join_token.join(sentence[pos_max[1]:]))
            # sent0 = sentence[:pos_min[0]]
            # ent0 = sentence[pos_min[0]:pos_min[1]]
            # sent1 = sentence[pos_min[1]:pos_max[0]]
            # ent1 = sentence[pos_max[0]:pos_max[1]]
            # sent2 = sentence[pos_max[1]:]

        if self.mask_entity:
            ent0 = ['[unused5]'] if not rev else ['[unused6]']
            ent1 = ['[unused6]'] if not rev else ['[unused5]']
        else:
            ent0 = ['[unused1]'] + ent0 + ['[unused2]'] if not rev else ['[unused3]'] + ent0 + ['[unused4]']
            ent1 = ['[unused3]'] + ent1 + ['[unused4]'] if not rev else ['[unused1]'] + ent1 + ['[unused2]']

        # get tokens index
        tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]']
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        avai_len = len(indexed_tokens)

        # Padding
        if self.blank_padding:
            if len(indexed_tokens) <= self.max_length:
                padding_idx = self.tokenizer.convert_tokens_to_ids('[PAD]')
                indexed_tokens.extend([padding_idx] * (self.max_length - len(indexed_tokens)))
            else:
                indexed_tokens[self.max_length - 1] = indexed_tokens[-1]
                indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask



class BERTWithDSPEncoder(BERTEncoder):
    def __init__(self, pretrain_path, bert_name='bert', max_length=256, max_dsp_path_length=15, dsp_tool='ddp', use_attention=True, blank_padding=True, mask_entity=False, compress_seq=False, language='en'):
        """
        Args:
            pretrain_path (str): path of pretrain model
            bert_name (str, optional): name of pretrained 'bert series' model. Defaults to 'bert'.
            max_length (int, optional): max_length of sequence. Defaults to 256.
            max_dsp_path_length (int, optional): 15 for ddp/stanza(true_max_len=12), 10 for ltp(true_max_len=9). Defaults to 15.
            dsp_tool (str, optional): DSP tool used: ltp, ddp or stanza. Defaults to 'ddp'.
            use_attention (bool, optional): whether use attention for dsp path feature, otherwise use maxpool. Defaults to True.
            blank_padding (bool, optional): whether pad sequence to the same length. Defaults to True.
            mask_entity (bool, optional): whether do mask for entity. Defaults to False.
            compress_seq (bool, optional): whether compress sequence. Defaults to False.

        Raises:
            NotImplementedError: DSP tool is not implemented.
        """
        super(BERTWithDSPEncoder, self).__init__(pretrain_path=pretrain_path, 
                                                max_length=max_length, 
                                                bert_name=bert_name,
                                                blank_padding=True, 
                                                mask_entity=False,
                                                language=language)
        self.max_dsp_path_length = max_dsp_path_length
        self.parser = None
        if self.max_dsp_path_length > 0:
            if dsp_tool == 'ddp':
                self.parser = DDP_Parse()
            elif dsp_tool == 'ltp':
                self.parser = LTP_Parse()
            elif dsp_tool == 'stanza':
                self.parser = Stanza_Parse()
            else:
                raise NotImplementedError(f'{dsp_tool} DSP tool is not implemented')
        bert_hidden_size = self.bert.config.hidden_size
        self.hidden_size = self.bert.config.hidden_size * 2

        self.bilstm = nn.LSTM(input_size=bert_hidden_size, 
                            hidden_size=bert_hidden_size, 
                            num_layers=1, 
                            bidirectional=False, 
                            batch_first=True)
        self.compress_seq = compress_seq
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        # attention
        self.use_attention = use_attention
        # self.attention_map = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.query = nn.Linear(bert_hidden_size, 1)

    def dsp_encode(self, hidden, dsp_path, dsp_path_length):
        # dsp_rep = torch.gather(hidden, 1, dsp_path.unsqueeze(-1).repeat(1, 1, hidden.size(-1)))
        dsp_rep = torch.stack([hidden[i, dsp_path[i]] for i in range(hidden.size(0))], dim=0) # (B, S, d)
        ## head entity dsp path representation
        if self.compress_seq:
            sorted_length_indices = dsp_path_length.argsort(descending=True) # (B,)
            unsorted_length_indices = sorted_length_indices.argsort(descending=False) # (B,)
            dsp_rep_packed = pack_padded_sequence(dsp_rep[sorted_length_indices], dsp_path_length[sorted_length_indices], batch_first=True)
            dsp_hidden_packed, _ = self.bilstm(dsp_rep_packed)
            dsp_hidden, _ = pad_packed_sequence(dsp_hidden_packed, batch_first=True) # (B, S, d)
            dsp_hidden = dsp_hidden[unsorted_length_indices] # (B, S, d)
        else:
            dsp_hidden, _ = self.bilstm(dsp_rep) # (B, S, d)
        return dsp_hidden

    def attention(self, dsp_hidden, dsp_path_length):
        # dsp_hidden = torch.tanh(self.attention_map(dsp_hidden))
        attention_score = self.query(dsp_hidden).squeeze(dim=-1) # (B, S)
        for i in range(attention_score.size(0)):
            attention_score[i, dsp_path_length[i]:] = 1e-9
        attention_distribution = F.softmax(attention_score, dim=-1) # (B, S)
        attention_output = torch.matmul(attention_distribution.unsqueeze(dim=1), dsp_hidden).squeeze(dim=1) # (B, d)
        return attention_output

    def forward(self, seqs, att_mask, ent_h_path, ent_t_path, ent_h_length, ent_t_length):
        """
        Args:
            seqs: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        if 'roberta' in self.bert_name:
            hidden, pooler_out = self.bert(seqs, attention_mask=att_mask) # clue-roberta
        else:
            hidden, pooler_out = self.bert(seqs, attention_mask=att_mask)
        # dsp encode, get dsp hidden
        ent_h_dsp_hidden = self.dsp_encode(hidden, ent_h_path, ent_h_length) # (B, S, d)
        ent_t_dsp_hidden = self.dsp_encode(hidden, ent_t_path, ent_t_length) # (B, S, d)
        # use attention or max pool to gather sequence hidden state
        if self.use_attention:
            ent_h_dsp_hidden = self.attention(ent_h_dsp_hidden, ent_h_length) # (B, d)
            ent_t_dsp_hidden = self.attention(ent_t_dsp_hidden, ent_t_length) # (B, d)
        else:
            for i in range(hidden.size(0)):
                ent_h_dsp_hidden[i][0] = ent_h_dsp_hidden[i][:ent_h_length[i]].max(dim=0)[0]
            ent_h_dsp_hidden = ent_h_dsp_hidden[:, 0] # (B, d)
            for i in range(hidden.size(0)):
                ent_t_dsp_hidden[i][0] = ent_t_dsp_hidden[i][:ent_t_length[i]].max(dim=0)[0]
            ent_t_dsp_hidden = ent_t_dsp_hidden[:, 0] # (B, d)
        ## cat head and tail representation
        ent_dsp_hidden = torch.add(ent_h_dsp_hidden, ent_t_dsp_hidden) # (B, d)
        # ent_dsp_hidden = torch.cat([ent_h_dsp_hidden, ent_t_dsp_hidden], dim=-1) # (B, 2d)

        joint_feature = torch.cat([pooler_out, ent_dsp_hidden], dim=-1)
        joint_feature = torch.tanh(self.linear(joint_feature))
        # joint_feature = self.linear(joint_feature)

        return joint_feature

    def tokenize(self, item):
        """
        Args:
            item: data instance containing 'text' / 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        ret_items = super(BERTWithDSPEncoder, self).tokenize(item)
        if self.parser is not None:
            # shortest dependency path
            ent_h_path, ent_t_path = self.parser.parse(self.sentence, item['h'], item['t'])
            ent_h_length = len(ent_h_path)
            ent_t_length = len(ent_t_path)
            if self.blank_padding:
                if ent_h_length <= self.max_dsp_path_length:
                    while len(ent_h_path) < self.max_dsp_path_length:
                        ent_h_path.append(-1)
                ent_h_path = ent_h_path[:self.max_dsp_path_length]
                if ent_t_length <= self.max_dsp_path_length:
                    while len(ent_t_path) < self.max_dsp_path_length:
                        ent_t_path.append(-1)
                ent_t_path = ent_t_path[:self.max_dsp_path_length]
            ent_h_path = torch.tensor([ent_h_path]).long() + 1
            ent_t_path = torch,tensor([ent_t_path]).long() + 1
            ent_h_length = torch.tensor([min(ent_h_length, self.max_dsp_path_length)]).long()
            ent_t_length = torch.tensor([min(ent_t_length, self.max_dsp_path_length)]).long()
            ret_items += (ent_h_path, ent_t_path, ent_h_length, ent_t_length)
        return ret_items



class BERTEntityEncoder(nn.Module):
    def __init__(self, pretrain_path, bert_name='bert', max_length=256, tag2id=None, blank_padding=True, mask_entity=False, language='en'):
        """
        Args:
            pretrain_path (str): path of pretrain model
            bert_name (str, optional): name of pretrained 'bert series' model. Defaults to 'bert'.
            max_length (int, optional): max_length of sequence. Defaults to 256.
            tag2id (dict, optional): entity type to id dictionary. Defaults to None.
            blank_padding (bool, optional): whether pad sequence to the same length. Defaults to True.
            mask_entity (bool, optional): whether do mask for entity. Defaults to False.
        
        Raises:
            NotImplementedError: bert pretrained model is not implemented.
        """
        super().__init__()
        self.language = language
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.mask_entity = mask_entity
        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert_name = bert_name
        if 'albert' in bert_name:
            if self.language == 'zh':
                self.bert = AlbertModel.from_pretrained(pretrain_path) # clue
                self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
            else:
                self.bert = AlbertModel.from_pretrained(pretrain_path)
                self.tokenizer = AlbertTokenizer.from_pretrained(pretrain_path)
        elif 'roberta' in bert_name:
            if self.language == 'zh':
                self.bert = BertModel.from_pretrained(pretrain_path) # clue, hfl
                self.tokenizer = BertTokenizer.from_pretrained(pretrain_path) # clue, hfl
            else:
                self.bert = RobertaModel.from_pretrained(pretrain_path)
                self.tokenizer = RobertaTokenizer.from_pretrained(pretrain_path)
        elif 'bert' in bert_name:
            # self.bert = AutoModelForMaskedLM.from_pretrained(pretrain_path, output_hidden_states=True)
            self.bert = BertModel.from_pretrained(pretrain_path)
            self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        else:
            raise NotImplementedError(f'{bert_name} pretrained model is not implemented!')
        # add possible missed tokens in vocab.txt
        num_added_tokens = self.tokenizer.add_tokens(['“', '”', '—'])
        logging.info(f"we have added {num_added_tokens} tokens ['“', '”', '—']")
        self.bert.resize_token_embeddings(len(self.tokenizer))

        bert_hidden_size = self.bert.config.hidden_size
        self.hidden_size = bert_hidden_size * 2
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        # for type boarder
        self.tag2id = tag2id


    def forward(self, seqs, pos1, pos2, att_mask):
        """
        Args:
            seqs: (B, L), index of tokens
            pos1: (B, 1), position of the head entity starter
            pos2: (B, 1), position of the tail entity starter
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Returns:
            (B, 2H), representations for sentences
        """
        if 'roberta' in self.bert_name:
            # hidden = self.bert(seqs, attention_mask=att_mask)[1][1] # hfl roberta
            hidden, _ = self.bert(seqs, attention_mask=att_mask) # clue-roberta
        else:
            hidden, _ = self.bert(seqs, attention_mask=att_mask)
        # Get entity start hidden state
        onehot_head = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_tail = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_head = onehot_head.scatter_(1, pos1, 1)
        onehot_tail = onehot_tail.scatter_(1, pos2, 1)
        head_hidden = (onehot_head.unsqueeze(2) * hidden).sum(1)  # (B, H)
        tail_hidden = (onehot_tail.unsqueeze(2) * hidden).sum(1)  # (B, H)
        rep_out = torch.cat([head_hidden, tail_hidden], 1)  # (B, 3H)
        rep_out = self.linear(rep_out)
        return rep_out

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
        self.sentence = sentence
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

        join_token = ' ' if self.language == 'en' else ''
        if not is_token:
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
        else:
            sent0 = self.tokenizer.tokenize(join_token.join(sentence[:pos_min[0]]))
            ent0 = self.tokenizer.tokenize(join_token.join(sentence[pos_min[0]:pos_min[1]]))
            sent1 = self.tokenizer.tokenize(join_token.join(sentence[pos_min[1]:pos_max[0]]))
            ent1 = self.tokenizer.tokenize(join_token.join(sentence[pos_max[0]:pos_max[1]]))
            sent2 = self.tokenizer.tokenize(join_token.join(sentence[pos_max[1]:]))
            # sent0 = sentence[:pos_min[0]]
            # ent0 = sentence[pos_min[0]:pos_min[1]]
            # sent1 = sentence[pos_min[1]:pos_max[0]]
            # ent1 = sentence[pos_max[0]:pos_max[1]]
            # sent2 = sentence[pos_max[1]:]

        if self.mask_entity:
            ent0 = ['[unused1]'] if not rev else ['[unused2]']
            ent1 = ['[unused2]'] if not rev else ['[unused1]']
        else:
            if self.tag2id:
                tag_head = item['h']['entity']
                tag_tail = item['t']['entity']
                if not rev:
                    ent0_left_boundary = ['[unused{}]'.format(3 + self.tag2id[tag_head] * 2)]
                    ent0_right_boundary = ['[unused{}]'.format(3 + self.tag2id[tag_head] * 2 + 1)]
                    ent1_left_boundary = ['[unused{}]'.format(3 + self.tag2id[tag_tail] * 2 + len(self.tag2id))]
                    ent1_right_boundary = ['[unused{}]'.format(3 + self.tag2id[tag_tail] * 2 + 1 + len(self.tag2id))]
                else:
                    ent0_left_boundary = ['[unused{}]'.format(3 + self.tag2id[tag_tail] * 2 + len(self.tag2id))]
                    ent0_right_boundary = ['[unused{}]'.format(3 + self.tag2id[tag_tail] * 2 + 1 + len(self.tag2id))]
                    ent1_left_boundary = ['[unused{}]'.format(3 + self.tag2id[tag_head] * 2)]
                    ent1_right_boundary = ['[unused{}]'.format(3 + self.tag2id[tag_head] * 2 + 1)]

                ent0 = ent0_left_boundary + ent0 + ent0_right_boundary
                ent1 = ent1_left_boundary + ent1 + ent1_right_boundary
            else:
                ent0 = ['[unused3]'] + ent0 + ['[unused4]'] if not rev else ['[unused5]'] + ent0 + ['[unused6]']
                ent1 = ['[unused5]'] + ent1 + ['[unused6]'] if not rev else ['[unused3]'] + ent1 + ['[unused4]']

        # get tokens index and entity position
        self.tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]']
        pos1_1 = 1 + len(sent0) if not rev else 1 + len(sent0 + ent0 + sent1)
        pos1_2 = pos1_1 + len(ent0) - 1 if not rev else pos1_1 + len(ent1) - 1
        pos2_1 = 1 + len(sent0 + ent0 + sent1) if not rev else 1 + len(sent0)
        pos2_2 = pos2_1 + len(ent1) - 1 if not rev else pos2_1 + len(ent0) - 1
        pos1_1 = torch.tensor([[min(self.max_length - 1, pos1_1)]]).long()
        pos1_2 = torch.tensor([[min(self.max_length - 1, pos1_2)]]).long()
        pos2_1 = torch.tensor([[min(self.max_length - 1, pos2_1)]]).long()
        pos2_2 = torch.tensor([[min(self.max_length - 1, pos2_2)]]).long()
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(self.tokens)
        avai_len = len(indexed_tokens) # 序列实际长度

        # Padding
        if self.blank_padding:
            if len(indexed_tokens) <= self.max_length:
                padding_idx = self.tokenizer.convert_tokens_to_ids('[PAD]')
                indexed_tokens.extend([padding_idx] * (self.max_length - len(indexed_tokens)))
            else:
                indexed_tokens[self.max_length - 1] = indexed_tokens[-1]
                indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        return indexed_tokens, pos1_1, pos2_1, pos1_2, pos2_2, att_mask



class BERTEntityWithContextEncoder(BERTEntityEncoder):
    def __init__(self, pretrain_path, bert_name='bert', max_length=256, tag2id=None, use_attention4context=True, blank_padding=True, mask_entity=False, language='en'):
        """
        Args:
            pretrain_path (str): path of pretrain model
            bert_name (str, optional): name of pretrained 'bert series' model. Defaults to 'bert'.
            max_length (int, optional): max_length of sequence. Defaults to 256.
            tag2id (dict, optional): entity type to id dictionary. Defaults to None.
            use_attention4context (bool, optional): whether use attention for context. Defaults to True.
            blank_padding (bool, optional): whether pad sequence to the same length. Defaults to True.
            mask_entity (bool, optional): whether do mask for entity. Defaults to False.
        
        Raises:
            NotImplementedError: bert pretrained model is not implemented.
        """
        super(BERTEntityWithContextEncoder, self).__init__(pretrain_path=pretrain_path, 
                                                    bert_name=bert_name,
                                                    max_length=max_length, 
                                                    tag2id=tag2id, 
                                                    blank_padding=blank_padding, 
                                                    mask_entity=mask_entity,
                                                    language=language)
        bert_hidden_size = self.bert.config.hidden_size
        self.hidden_size = bert_hidden_size * 3
        self.use_attention4context = use_attention4context
        if self.use_attention4context:
            self.context_query = nn.Linear(bert_hidden_size, 1)
        else:
            self.conv = nn.Conv2d(1, bert_hidden_size, kernel_size=(5, bert_hidden_size))  # add a convolution layer to extract the global information of sentence
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)


    def attention(self, query, hidden, seq_length):
        # dsp_hidden = torch.tanh(self.attention_map(dsp_hidden))
        attention_score = query(hidden).squeeze(dim=-1) # (B, S)
        for i in range(hidden.size(0)):
            attention_score[i, seq_length[i]:] = 1e-9
        attention_distribution = F.softmax(attention_score, dim=-1) # (B, S)
        attention_output = torch.matmul(attention_distribution.unsqueeze(dim=1), hidden).squeeze(dim=1) # (B, d)
        return attention_output


    def forward(self, seqs, pos1, pos2, att_mask):
        """
        Args:
            seqs: (B, L), index of tokens
            pos1: (B, 1), position of the head entity starter
            pos2: (B, 1), position of the tail entity starter
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Returns:
            (B, 2H), representations for sentences
        """
        if 'roberta' in self.bert_name:
            # hidden = self.bert(seqs, attention_mask=att_mask)[1][1] # hfl roberta
            hidden, _ = self.bert(seqs, attention_mask=att_mask) # clue-roberta
        else:
            hidden, _ = self.bert(seqs, attention_mask=att_mask)
        # Get entity start hidden state
        onehot_head = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_tail = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_head = onehot_head.scatter_(1, pos1, 1)
        onehot_tail = onehot_tail.scatter_(1, pos2, 1)
        head_hidden = (onehot_head.unsqueeze(2) * hidden).sum(1)  # (B, H)
        tail_hidden = (onehot_tail.unsqueeze(2) * hidden).sum(1)  # (B, H)
        # x = torch.cat([head_hidden, tail_hidden], 1)  # (B, 2H)
        if self.use_attention4context:
            context_hidden = self.attention(self.context_query, hidden, att_mask.sum(dim=-1)) # (B, d)
        else:
            context_conv = self.conv(hidden.unsqueeze(1)).squeeze(3) # (B, d, S)
            context_hidden = F.relu(F.max_pool1d(context_conv, 
                                    context_conv.size(2)).squeeze(2)) # (B, d), maxpool->relu is more efficient than relu->maxpool
        rep_out = torch.cat([head_hidden, tail_hidden, context_hidden], 1)  # (B, 3H)
        rep_out = self.linear(rep_out)
        return rep_out



class BERTEntityWithDSPEncoder(BERTEntityEncoder):
    def __init__(self, pretrain_path, bert_name='bert', max_length=256, max_dsp_path_length=15, dsp_tool='ddp', tag2id=None, 
                use_attention4dsp=True, blank_padding=True, mask_entity=False, compress_seq=False, language='en'):
        """
        Args:
            pretrain_path (str): path of pretrain model
            bert_name (str, optional): name of pretrained 'bert series' model. Defaults to 'bert'.
            max_length (int, optional): max_length of sequence. Defaults to 256.
            max_dsp_path_length (int, optional): 15 for ddp/stanza(true_max_len=12), 10 for ltp(true_max_len=9). Defaults to 15.
            dsp_tool (str, optional): DSP tool used: ltp, ddp or stanza. Defaults to 'ddp'.
            tag2id (dict, optional): entity type to id dictionary. Defaults to None.
            use_attention4dsp (bool, optional): whether use attention for dsp path feature, otherwise use maxpool. Defaults to True.
            blank_padding (bool, optional): whether pad sequence to the same length. Defaults to True.
            mask_entity (bool, optional): whether do mask for entity. Defaults to False.
            compress_seq (bool, optional): whether compress sequence. Defaults to False.

        Raises:
            NotImplementedError: DSP tool is not implemented.
        """
        super(BERTEntityWithDSPEncoder, self).__init__(pretrain_path=pretrain_path, 
                                                    bert_name=bert_name,
                                                    max_length=max_length, 
                                                    tag2id=tag2id, 
                                                    blank_padding=blank_padding, 
                                                    mask_entity=mask_entity,
                                                    language=language)
        self.max_dsp_path_length = max_dsp_path_length
        self.parser = None
        if self.max_dsp_path_length > 0:
            if dsp_tool == 'ddp':
                self.parser = DDP_Parse()
            elif dsp_tool == 'ltp':
                self.parser = LTP_Parse()
            elif dsp_tool == 'stanza':
                self.parser = Stanza_Parse()
            else:
                raise NotImplementedError(f'{dsp_tool} DSP tool is not implemented')
        bert_hidden_size = self.bert.config.hidden_size
        self.bilstm = nn.LSTM(input_size=bert_hidden_size, 
                            hidden_size=bert_hidden_size, 
                            num_layers=1, 
                            bidirectional=False, 
                            batch_first=True)
        self.compress_seq = compress_seq
        self.hidden_size = bert_hidden_size * 4
        # output map
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        # attention
        self.use_attention4dsp = use_attention4dsp
        # self.attention_map = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.dsp_query = nn.Linear(bert_hidden_size, 1)

    def dsp_encode(self, hidden, dsp_path, dsp_path_length):
        # dsp_rep = torch.gather(hidden, 1, dsp_path.unsqueeze(-1).repeat(1, 1, hidden.size(-1)))
        # dsp_rep = torch.gather(hidden, 1, dsp_path.unsqueeze(-1).expand(*dsp_path.size(), hidden.size(-1)))
        dsp_rep = torch.stack([hidden[i, dsp_path[i]] for i in range(hidden.size(0))], dim=0) # (B, S, d)
        ## head entity dsp path representation
        if self.compress_seq:
            sorted_length_indices = dsp_path_length.argsort(descending=True) # (B,)
            unsorted_length_indices = sorted_length_indices.argsort(descending=False) # (B,)
            dsp_rep_packed = pack_padded_sequence(dsp_rep[sorted_length_indices], dsp_path_length[sorted_length_indices], batch_first=True)
            dsp_hidden_packed, _ = self.bilstm(dsp_rep_packed)
            dsp_hidden, _ = pad_packed_sequence(dsp_hidden_packed, batch_first=True) # (B, S, d)
            dsp_hidden = dsp_hidden[unsorted_length_indices] # (B, S, d)
        else:
            dsp_hidden, _ = self.bilstm(dsp_rep) # (B, S, d)
        return dsp_hidden

    def attention(self, query, hidden, seq_length):
        # dsp_hidden = torch.tanh(self.attention_map(dsp_hidden))
        attention_score = query(hidden).squeeze(dim=-1) # (B, S)
        for i in range(hidden.size(0)):
            attention_score[i, seq_length[i]:] = 1e-9
        attention_distribution = F.softmax(attention_score, dim=-1) # (B, S)
        attention_output = torch.matmul(attention_distribution.unsqueeze(dim=1), hidden).squeeze(dim=1) # (B, d)
        return attention_output

    def forward(self, seqs, pos1, pos2, att_mask, ent_h_path, ent_t_path, ent_h_length, ent_t_length):
        """
        Args:
            seqs: (B, L), index of tokens
            pos1: (B, 1), position of the head entity starter
            pos2: (B, 1), position of the tail entity starter
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Returns:
            (B, 2H), representations for sentences
        """
        if 'roberta' in self.bert_name:
            hidden, pooler_out = self.bert(seqs, attention_mask=att_mask) # clue-roberta
        else:
            hidden, pooler_out = self.bert(seqs, attention_mask=att_mask)
        # Get entity start hidden state
        onehot_head = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_tail = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_head = onehot_head.scatter_(1, pos1, 1)
        onehot_tail = onehot_tail.scatter_(1, pos2, 1)
        head_hidden = torch.matmul(onehot_head.unsqueeze(1), hidden).squeeze(1)  # (B, H)
        tail_hidden = torch.matmul(onehot_tail.unsqueeze(1), hidden).squeeze(1)  # (B, H)

        # dsp encode, get dsp hidden
        ent_h_dsp_hidden = self.dsp_encode(hidden, ent_h_path, ent_h_length) # (B, S, d)
        ent_t_dsp_hidden = self.dsp_encode(hidden, ent_t_path, ent_t_length) # (B, S, d)
        # use attention or max pool to gather sequence hidden state
        if self.use_attention4dsp:
            ent_h_dsp_hidden = self.attention(self.dsp_query, ent_h_dsp_hidden, ent_h_length) # (B, d)
            ent_t_dsp_hidden = self.attention(self.dsp_query, ent_t_dsp_hidden, ent_t_length) # (B, d)
        else:
            for i in range(hidden.size(0)):
                ent_h_dsp_hidden[i][0] = ent_h_dsp_hidden[i][:ent_h_length[i]].max(dim=0)[0]
            ent_h_dsp_hidden = ent_h_dsp_hidden[:, 0] # (B, d)
            for i in range(hidden.size(0)):
                ent_t_dsp_hidden[i][0] = ent_t_dsp_hidden[i][:ent_t_length[i]].max(dim=0)[0]
            ent_t_dsp_hidden = ent_t_dsp_hidden[:, 0] # (B, d)
        ## cat head and tail representation
        # dsp_hidden = torch.add(ent_h_dsp_hidden, ent_t_dsp_hidden) # (B, d)
        dsp_hidden = torch.cat([ent_h_dsp_hidden, ent_t_dsp_hidden], dim=-1) # (B, 2d)

        # gather all features
        rep_out = torch.cat([head_hidden, tail_hidden, dsp_hidden], dim=-1)  # (B, 4H)
        # rep_out = torch.cat([head_hidden, tail_hidden, pooler_output, dsp_hidden], dim=-1)  # (B, 4H)
        rep_out = torch.tanh(self.linear(rep_out))
        # rep_out = self.linear(rep_out)

        return rep_out

    def tokenize(self, item):
        """
        Args:
            item: data instance containing 'text' / 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        ret_items = super(BERTEntityWithDSPEncoder, self).tokenize(item)
        if self.parser is not None:
            ent_h_pos_1 = torch.min(ret_items[1], ret_items[2]).item()
            ent_h_pos_2 = torch.min(ret_items[3], ret_items[4]).item()
            ent_t_pos_1 = torch.max(ret_items[1], ret_items[2]).item()
            ent_t_pos_2 = torch.max(ret_items[3], ret_items[4]).item()
            # shortest dependency path
            for i, t in enumerate(self.tokens[1:-1]):
                if t.startswith('##'):
                    self.tokens[i + 1] = t[2:]
            ent_h_path, ent_t_path = self.parser.parse(self.sentence, item['h'], item['t'], self.tokens[1:-1])
            ent_h_length = len(ent_h_path)
            ent_t_length = len(ent_t_path)
            if self.blank_padding:
                ent_h_path = ent_h_path[:self.max_dsp_path_length]
                ent_t_path = ent_t_path[:self.max_dsp_path_length]
            # move parsed token pos if entity type embedded
            for i, pos in enumerate(ent_h_path):
                if pos >= ent_h_pos_1:
                    pos += 1
                if pos >= ent_h_pos_2:
                    pos += 1
                if pos >= ent_t_pos_1:
                    pos += 1
                if pos >= ent_t_pos_2:
                    pos += 1
                ent_h_path[i] = pos
            for i, pos in enumerate(ent_t_path):
                if pos >= ent_h_pos_1:
                    pos += 1
                if pos >= ent_h_pos_2:
                    pos += 1
                if pos >= ent_t_pos_1:
                    pos += 1
                if pos >= ent_t_pos_2:
                    pos += 1
                ent_t_path[i] = pos
            if self.blank_padding:
                if ent_h_length < self.max_dsp_path_length:
                    ent_h_path.extend([0] * (self.max_dsp_path_length - len(ent_h_path)))
                if ent_t_length < self.max_dsp_path_length:
                    ent_t_path.extend([0] * (self.max_dsp_path_length - len(ent_t_path)))
            ent_h_path = torch.tensor([ent_h_path]).long()
            ent_t_path = torch.tensor([ent_t_path]).long()
            ent_h_length = torch.tensor([min(ent_h_length, self.max_dsp_path_length)]).long()
            ent_t_length = torch.tensor([min(ent_t_length, self.max_dsp_path_length)]).long()
            ret_items += (ent_h_path, ent_t_path, ent_h_length, ent_t_length)
        return ret_items



class BERTEntityWithContextDSPEncoder(BERTEntityWithDSPEncoder):
    def __init__(self, pretrain_path, bert_name='bert', max_length=256, max_dsp_path_length=15, dsp_tool='ddp', tag2id=None, 
                use_attention4dsp=True, use_attention4context=True, blank_padding=True, mask_entity=False, compress_seq=False, language='en'):
        """
        Args:
            pretrain_path (str): path of pretrain model
            bert_name (str, optional): name of pretrained 'bert series' model. Defaults to 'bert'.
            max_length (int, optional): max_length of sequence. Defaults to 256.
            max_dsp_path_length (int, optional): 15 for ddp/stanza(true_max_len=12), 10 for ltp(true_max_len=9). Defaults to 15.
            dsp_tool (str, optional): DSP tool used: ltp, ddp or stanza. Defaults to 'ddp'.
            tag2id (dict, optional): entity type to id dictionary. Defaults to None.
            use_attention4dsp (bool, optional): whether use attention for dsp path feature, otherwise use maxpool. Defaults to True.
            use_attention4context (bool, optional): whether use attention for context. Defaults to True.
            blank_padding (bool, optional): whether pad sequence to the same length. Defaults to True.
            mask_entity (bool, optional): whether do mask for entity. Defaults to False.
            compress_seq (bool, optional): whether compress sequence. Defaults to False.

        Raises:
            NotImplementedError: DSP tool is not implemented.
        """
        super(BERTEntityWithContextDSPEncoder, self).__init__(pretrain_path=pretrain_path, 
                                                    bert_name=bert_name,
                                                    max_length=max_length, 
                                                    max_dsp_path_length=max_dsp_path_length,
                                                    dsp_tool=dsp_tool,
                                                    tag2id=tag2id, 
                                                    use_attention4dsp=use_attention4dsp,
                                                    blank_padding=blank_padding, 
                                                    mask_entity=mask_entity,
                                                    compress_seq=compress_seq,
                                                    language=language)
        bert_hidden_size = self.bert.config.hidden_size
        self.hidden_size = bert_hidden_size * 5
        # output map
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        # conv
        self.use_attention4context = use_attention4context
        if self.use_attention4context:
            self.context_query = nn.Linear(bert_hidden_size, 1)
        else:
            self.conv = nn.Conv2d(1, bert_hidden_size, kernel_size=(5, bert_hidden_size))  # add a convolution layer to extract the global information of sentence

    def forward(self, seqs, pos1, pos2, att_mask, ent_h_path, ent_t_path, ent_h_length, ent_t_length):
        """
        Args:
            seqs: (B, L), index of tokens
            pos1: (B, 1), position of the head entity starter
            pos2: (B, 1), position of the tail entity starter
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Returns:
            (B, 2H), representations for sentences
        """
        if 'roberta' in self.bert_name:
            hidden, pooler_out = self.bert(seqs, attention_mask=att_mask) # clue-roberta
        else:
            hidden, pooler_out = self.bert(seqs, attention_mask=att_mask)
        # Get entity start hidden state
        onehot_head = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_tail = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_head = onehot_head.scatter_(1, pos1, 1)
        onehot_tail = onehot_tail.scatter_(1, pos2, 1)
        head_hidden = torch.matmul(onehot_head.unsqueeze(1), hidden).squeeze(1)  # (B, H)
        tail_hidden = torch.matmul(onehot_tail.unsqueeze(1), hidden).squeeze(1)  # (B, H)

        # get context representation
        if self.use_attention4context:
            context_hidden = self.attention(self.context_query, hidden, att_mask.sum(dim=-1)) # (B, d)
        else:
            context_conv = self.conv(hidden.unsqueeze(1)).squeeze(3) # (B, d, S)
            context_hidden = F.relu(F.max_pool1d(context_conv, 
                                    context_conv.size(2)).squeeze(2)) # (B, d), maxpool->relu is more efficient than relu->maxpool

        # dsp encode, get dsp hidden
        ent_h_dsp_hidden = self.dsp_encode(hidden, ent_h_path, ent_h_length) # (B, S, d)
        ent_t_dsp_hidden = self.dsp_encode(hidden, ent_t_path, ent_t_length) # (B, S, d)
        # use attention or max pool to gather sequence hidden state
        if self.use_attention4dsp:
            ent_h_dsp_hidden = self.attention(self.dsp_query, ent_h_dsp_hidden, ent_h_length) # (B, d)
            ent_t_dsp_hidden = self.attention(self.dsp_query, ent_t_dsp_hidden, ent_t_length) # (B, d)
        else:
            for i in range(hidden.size(0)):
                ent_h_dsp_hidden[i][0] = ent_h_dsp_hidden[i][:ent_h_length[i]].max(dim=0)[0]
            ent_h_dsp_hidden = ent_h_dsp_hidden[:, 0] # (B, d)
            for i in range(hidden.size(0)):
                ent_t_dsp_hidden[i][0] = ent_t_dsp_hidden[i][:ent_t_length[i]].max(dim=0)[0]
            ent_t_dsp_hidden = ent_t_dsp_hidden[:, 0] # (B, d)
        ## cat head and tail representation
        # dsp_hidden = torch.add(ent_h_dsp_hidden, ent_t_dsp_hidden) # (B, d)
        dsp_hidden = torch.cat([ent_h_dsp_hidden, ent_t_dsp_hidden], dim=-1) # (B, 2d)

        # gather all features
        rep_out = torch.cat([head_hidden, tail_hidden, context_hidden, dsp_hidden], dim=-1)  # (B, 4H)
        # rep_out = torch.cat([head_hidden, tail_hidden, pooler_output, dsp_hidden], dim=-1)  # (B, 4H)
        rep_out = torch.tanh(self.linear(rep_out))
        # rep_out = self.linear(rep_out)

        return rep_out

    

class RBERTEncoder(nn.Module):
    def __init__(self, pretrain_path, bert_name='bert', max_length=256, tag2id=None, blank_padding=True, mask_entity=False, language='en'):
        """
        Args:
            pretrain_path (str): path of pretrain model
            bert_name (str, optional): name of pretrained 'bert series' model. Defaults to 'bert'.
            max_length (int, optional): max_length of sequence. Defaults to 256.
            tag2id (dict, optional): entity type to id dictionary. Defaults to None.
            blank_padding (bool, optional): whether pad sequence to the same length. Defaults to True.
            mask_entity (bool, optional): whether do mask for entity. Defaults to False.
        
        Raises:
            NotImplementedError: bert pretrained model is not implemented.
        """
        super().__init__()
        self.language = language
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.mask_entity = mask_entity
        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert_name = bert_name
        if 'albert' in bert_name:
            self.bert = AlbertModel.from_pretrained(pretrain_path) # clue
            self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        if 'roberta' in bert_name:
            # self.bert = AutoModelForMaskedLM.from_pretrained(pretrain_path, output_hidden_states=True) # hfl
            # self.tokenizer = AutoTokenizer.from_pretrained(pretrain_path) # hfl
            self.bert = BertModel.from_pretrained(pretrain_path) # clue
            self.tokenizer = BertTokenizer.from_pretrained(pretrain_path) # clue
        elif 'bert' in bert_name:
            # self.bert = AutoModelForMaskedLM.from_pretrained(pretrain_path, output_hidden_states=True)
            self.bert = BertModel.from_pretrained(pretrain_path)
            self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        else:
            raise NotImplementedError(f'{bert_name} pretrained model is not implemented!')
        # add possible missed tokens in vocab.txt
        num_added_tokens = self.tokenizer.add_tokens(['“', '”', '—'])
        logging.info(f"we have added {num_added_tokens} tokens ['“', '”', '—']")
        self.bert.resize_token_embeddings(len(self.tokenizer))

        bert_hidden_size = self.bert.config.hidden_size
        self.hidden_size = bert_hidden_size * 3
        # self.conv = nn.Conv2d(1, bert_hidden_size, kernel_size=(5, bert_hidden_size))  # add a convolution layer to extract the global information of sentence
        self.w_ent = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.w_cls = nn.Linear(bert_hidden_size, bert_hidden_size)
        # for type boarder
        self.tag2id = tag2id

    def forward(self, seqs, ent1_span, ent2_span, att_mask):
        """
        Args:
            seqs: (B, L), index of tokens
            pos1: (B, 1), position of the head entity starter
            pos2: (B, 1), position of the tail entity starter
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Returns:
            (B, 2H), representations for sentences
        """
        if 'roberta' in self.bert_name:
            # hidden = self.bert(seqs, attention_mask=att_mask)[1][1] # hfl roberta
            hidden, _ = self.bert(seqs, attention_mask=att_mask) # clue-roberta
        else:
            hidden, _ = self.bert(seqs, attention_mask=att_mask)
        # Get entity start hidden state
        ent1_avg_hidden = torch.matmul(ent1_span.unsqueeze(1), hidden).squeeze(1) / torch.sum(ent1_span, dim=-1, keepdim=True) # (B, H)
        ent2_avg_hidden = torch.matmul(ent2_span.unsqueeze(1), hidden).squeeze(1) / torch.sum(ent2_span, dim=-1, keepdim=True) # (B, H)
        ent1_hidden = self.w_ent(torch.tanh(ent1_avg_hidden))
        ent2_hidden = self.w_ent(torch.tanh(ent2_avg_hidden))
        cls_hidden = self.w_cls(torch.tanh(hidden[:,0]))
        rep_out = torch.cat([ent1_hidden, ent2_hidden, cls_hidden], dim=-1)
        return rep_out

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
        self.sentence = sentence
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

        join_token = ' ' if self.language == 'en' else ''
        if not is_token:
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
        else:
            sent0 = self.tokenizer.tokenize(join_token.join(sentence[:pos_min[0]]))
            ent0 = self.tokenizer.tokenize(join_token.join(sentence[pos_min[0]:pos_min[1]]))
            sent1 = self.tokenizer.tokenize(join_token.join(sentence[pos_min[1]:pos_max[0]]))
            ent1 = self.tokenizer.tokenize(join_token.join(sentence[pos_max[0]:pos_max[1]]))
            sent2 = self.tokenizer.tokenize(join_token.join(sentence[pos_max[1]:]))
            # sent0 = sentence[:pos_min[0]]
            # ent0 = sentence[pos_min[0]:pos_min[1]]
            # sent1 = sentence[pos_min[1]:pos_max[0]]
            # ent1 = sentence[pos_max[0]:pos_max[1]]
            # sent2 = sentence[pos_max[1]:]

        if self.mask_entity:
            ent0 = ['[unused1]'] if not rev else ['[unused2]']
            ent1 = ['[unused2]'] if not rev else ['[unused1]']
        else:
            if self.tag2id:
                tag_head = item['h']['entity']
                tag_tail = item['t']['entity']
                if not rev:
                    ent0_left_boundary = ['[unused{}]'.format(3 + self.tag2id[tag_head] * 2)]
                    ent0_right_boundary = ['[unused{}]'.format(3 + self.tag2id[tag_head] * 2 + 1)]
                    ent1_left_boundary = ['[unused{}]'.format(3 + self.tag2id[tag_tail] * 2 + len(self.tag2id))]
                    ent1_right_boundary = ['[unused{}]'.format(3 + self.tag2id[tag_tail] * 2 + 1 + len(self.tag2id))]
                else:
                    ent0_left_boundary = ['[unused{}]'.format(3 + self.tag2id[tag_tail] * 2 + len(self.tag2id))]
                    ent0_right_boundary = ['[unused{}]'.format(3 + self.tag2id[tag_tail] * 2 + 1 + len(self.tag2id))]
                    ent1_left_boundary = ['[unused{}]'.format(3 + self.tag2id[tag_head] * 2)]
                    ent1_right_boundary = ['[unused{}]'.format(3 + self.tag2id[tag_head] * 2 + 1)]

                ent0 = ent0_left_boundary + ent0 + ent0_right_boundary
                ent1 = ent1_left_boundary + ent1 + ent1_right_boundary
            else:
                ent0 = ['[unused3]'] + ent0 + ['[unused4]'] if not rev else ['[unused5]'] + ent0 + ['[unused6]']
                ent1 = ['[unused5]'] + ent1 + ['[unused6]'] if not rev else ['[unused3]'] + ent1 + ['[unused4]']

        tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]']
        ent1_pos1 = 1 + len(sent0) if not rev else 1 + len(sent0 + ent0 + sent1)
        ent1_pos2 = 1 + len(sent0) + len(ent0) if not rev else 1 + len(sent0 + ent0 + sent1 + ent1)
        ent2_pos1 = 1 + len(sent0 + ent0 + sent1) if not rev else 1 + len(sent0)
        ent2_pos2 = 1 + len(sent0 + ent0 + sent1 + ent1) if not rev else 1 + len(sent0 + ent0)
        ent1_pos1 = min(self.max_length - 1, ent1_pos1)
        ent2_pos1 = min(self.max_length - 1, ent2_pos1)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        avai_len = len(indexed_tokens) # 序列实际长度

        # Position
        ent1_pos1 = torch.tensor([[ent1_pos1]]).long()
        ent1_pos2 = torch.tensor([[ent1_pos2]]).long()
        ent2_pos1 = torch.tensor([[ent2_pos1]]).long()
        ent2_pos2 = torch.tensor([[ent2_pos2]]).long()

        # Padding
        if self.blank_padding:
            if len(indexed_tokens) <= self.max_length:
                while len(indexed_tokens) < self.max_length:
                    indexed_tokens.append(0)  # 0 is id for [PAD]
            else:
                indexed_tokens[self.max_length - 1] = indexed_tokens[-1]
                indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1
        ent1_span = torch.zeros(indexed_tokens.size())
        ent1_span[0, ent1_pos1:ent1_pos2+1] = 1
        ent2_span = torch.zeros(indexed_tokens.size())
        ent2_span[0, ent2_pos1:ent2_pos2+1] = 1
        

        return indexed_tokens, ent1_span, ent2_span, att_mask
