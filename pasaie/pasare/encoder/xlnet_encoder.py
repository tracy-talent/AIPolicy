"""
 Author: liujian
 Date: 2021-02-21 19:33:13
 Last Modified by: liujian
 Last Modified time: 2021-02-21 19:33:13
"""

from .base_encoder import BaseEncoder
from ...utils.dependency_parse import DDP_Parse, LTP_Parse, Stanza_Parse

import logging
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from transformers import XLNetModel
from transformers import XLNetTokenizer



class XLNetEntityEncoder(nn.Module):
    def __init__(self, pretrain_path, max_length=256, tag2id=None, blank_padding=True, mask_entity=False, language='en'):
        """
        Args:
            pretrain_path (str): path of pretrain model
            max_length (int, optional): max_length of sequence. Defaults to 256.
            tag2id (dict, optional): entity type to id dictionary. Defaults to None.
            blank_padding (bool, optional): whether pad sequence to the same length. Defaults to True.
            mask_entity (bool, optional): whether do mask for entity. Defaults to False.
            language (str, optional): language of bert model, support 'zh' or 'en'.
        
        Raises:
            NotImplementedError: bert pretrained model is not implemented.
        """
        super().__init__()
        self.language = language
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.mask_entity = mask_entity
        logging.info('Loading XLNet pre-trained checkpoint.')
        self.xlnet = XLNetModel.from_pretrained(pretrain_path, mem_len=max_length)
        self.tokenizer = XLNetTokenizer.from_pretrained(pretrain_path)
        # add possible missed tokens in vocab.txt
        num_added_tokens = self.tokenizer.add_tokens(['“', '”', '—'])
        logging.info(f"we have added {num_added_tokens} tokens ['“', '”', '—']")
        self.xlnet.resize_token_embeddings(len(self.tokenizer))

        xlnet_hidden_size = self.xlnet.config.hidden_size
        self.hidden_size = xlnet_hidden_size * 2
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        # for type boarder
        self.tag2id = tag2id


    def forward(self, seqs, pos1, pos2, token_type_ids, att_mask):
        """
        Args:
            seqs: (B, L), index of tokens
            pos1: (B, 1), position of the head entity starter
            pos2: (B, 1), position of the tail entity starter
            token_type_ids: (B, L): type of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Returns:
            (B, 2H), representations for sentences
        """
        hidden, _ = self.xlnet(input_ids=seqs, token_type_ids=token_type_ids, attention_mask=att_mask)
        seq_len = torch.sum(att_mask, dim=-1)
        hidden_copy = hidden.clone()
        for i in range(seq_len.size(0)):
            hidden[i, :seq_len[i]] = hidden_copy[i, -seq_len[i]:]
            hidden[i, seq_len[i]:] = hidden_copy[i, :-seq_len[i]]
        hidden_copy = hidden_copy.detach().cpu()
        # Get entity start hidden state
        onehot_head = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_tail = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_head = onehot_head.scatter_(1, pos1, 1)
        onehot_tail = onehot_tail.scatter_(1, pos2, 1)
        head_hidden = (onehot_head.unsqueeze(2) * hidden).sum(1)  # (B, H)
        tail_hidden = (onehot_tail.unsqueeze(2) * hidden).sum(1)  # (B, H)
        rep_out = torch.cat([head_hidden, tail_hidden], 1)  # (B, 2H)
        rep_out = self.linear(rep_out)
        return rep_out


    def check_underline(self, tokens):
        if len(tokens) > 0:
            if tokens[0] == '▁':
                tokens.pop(0)


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
        if self.language == 'zh': # remove first token '▁' which get by chinese xlnet tokenizer
            if len(sent0) > 0:
                self.check_underline(ent0)
            self.check_underline(sent1)
            self.check_underline(ent1)
            self.check_underline(sent2)

        self.xlnet_tokens = sent0 + ent0 + sent1 + ent1 + sent2
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

        # get entity position and token index
        self.tokens = sent0 + ent0 + sent1 + ent1 + sent2 + ['<sep>', '<cls>']
        avai_len = len(self.tokens) # 序列实际长度
        pos1_1 = len(sent0) if not rev else len(sent0 + ent0 + sent1)
        pos1_2 = pos1_1 + len(ent0) if not rev else pos1_1 + len(ent1)
        pos2_1 = len(sent0 + ent0 + sent1) if not rev else len(sent0)
        pos2_2 = pos2_1 + len(ent1) if not rev else pos2_1 + len(ent0)
        pos1_1 = torch.tensor([[min(self.max_length - 1, pos1_1)]]).long()
        pos1_2 = torch.tensor([[min(self.max_length, pos1_2)]]).long()
        pos2_1 = torch.tensor([[min(self.max_length - 1, pos2_1)]]).long()
        pos2_2 = torch.tensor([[min(self.max_length, pos2_2)]]).long()

        # padding sequence
        if self.blank_padding:
            if len(self.tokens) < self.max_length:
                indexed_tokens = [self.tokenizer.convert_tokens_to_ids('<pad>')] * (self.max_length - len(self.tokens))
                token_type_ids = [3] * (self.max_length - len(self.tokens)) # 3 for <pad>
            else:
                indexed_tokens = []
                token_type_ids = []
        indexed_tokens.extend(self.tokenizer.convert_tokens_to_ids(self.tokens))
        token_type_ids.extend([0] * len(self.tokens))
        token_type_ids[-1] = 2 # 2 for <cls>
        if self.blank_padding:
            if len(indexed_tokens) > self.max_length:
                indexed_tokens = indexed_tokens[:self.max_length - 2] + indexed_tokens[-2:]
                token_type_ids = token_type_ids[:self.max_length - 1] + token_type_ids[-1:]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)
        token_type_ids = torch.tensor(token_type_ids).long().unsqueeze(0) # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, -avai_len:] = 1

        return indexed_tokens, pos1_1, pos2_1, pos1_2, pos2_2, token_type_ids, att_mask



class XLNetEntityWithContextEncoder(XLNetEntityEncoder):
    def __init__(self, pretrain_path, max_length=256, tag2id=None, use_attention4context=True, blank_padding=True, mask_entity=False, language='en'):
        """
        Args:
            pretrain_path (str): path of pretrain model
            max_length (int, optional): max_length of sequence. Defaults to 256.
            tag2id (dict, optional): entity type to id dictionary. Defaults to None.
            use_attention4context (bool, optional): whether use attention for context. Defaults to True.
            blank_padding (bool, optional): whether pad sequence to the same length. Defaults to True.
            mask_entity (bool, optional): whether do mask for entity. Defaults to False.
            language (str, optional): language of bert model, support 'zh' or 'en'.
        
        Raises:
            NotImplementedError: bert pretrained model is not implemented.
        """
        super(XLNetEntityWithContextEncoder, self).__init__(
            pretrain_path=pretrain_path, 
            max_length=max_length, 
            tag2id=tag2id, 
            blank_padding=blank_padding, 
            mask_entity=mask_entity,
            language=language
        )
        self.use_attention4context = use_attention4context
        xlnet_hidden_size = self.xlnet.config.hidden_size
        self.hidden_size = xlnet_hidden_size * 3
        if use_attention4context:
            self.context_query = nn.Linear(xlnet_hidden_size, 1)
        else:
            self.conv = nn.Conv1d(xlnet_hidden_size, xlnet_hidden_size, kernel_size=5)  # add a convolution layer to extract the global information of sentence
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)


    def attention(self, query, hidden, seq_length):
        # dsp_hidden = torch.tanh(self.attention_map(dsp_hidden))
        attention_score = query(hidden).squeeze(dim=-1) # (B, S)
        for i in range(hidden.size(0)):
            attention_score[i, seq_length[i]:] = 1e-9
        attention_distribution = F.softmax(attention_score, dim=-1) # (B, S)
        attention_output = torch.matmul(attention_distribution.unsqueeze(dim=1), hidden).squeeze(dim=1) # (B, d)
        return attention_output


    def forward(self, seqs, pos1, pos2, token_type_ids, att_mask):
        """
        Args:
            seqs: (B, L), index of tokens
            pos1: (B, 1), position of the head entity starter
            pos2: (B, 1), position of the tail entity starter
            token_type_ids: (B, L): type of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Returns:
            (B, 2H), representations for sentences
        """
        hidden, _ = self.xlnet(input_ids=seqs, token_type_ids=token_type_ids, attention_mask=att_mask)
        seq_len = torch.sum(att_mask, dim=-1)
        hidden_copy = hidden.clone()
        for i in range(seq_len.size(0)):
            hidden[i, :seq_len[i]] = hidden_copy[i, -seq_len[i]:]
            hidden[i, seq_len[i]:] = hidden_copy[i, :-seq_len[i]]
        hidden_copy = hidden_copy.detach().cpu()
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
            context_conv = self.conv(hidden.permute(0, 2, 1)) # (B, d, S)
            context_hidden = F.relu(F.max_pool1d(context_conv, 
                                    context_conv.size(2)).squeeze(2)) # (B, d), maxpool->relu is more efficient than relu->maxpool
        rep_out = torch.cat([head_hidden, tail_hidden, context_hidden], 1)  # (B, 3H)
        rep_out = self.linear(rep_out)
        return rep_out



class XLNetEntityWithDSPEncoder(XLNetEntityEncoder):
    def __init__(self, pretrain_path, max_length=256, max_dsp_path_length=15, dsp_tool='ddp', tag2id=None, 
                use_attention4dsp=True, blank_padding=True, mask_entity=False, compress_seq=False, language='en'):
        """
        Args:
            pretrain_path (str): path of pretrain model
            max_length (int, optional): max_length of sequence. Defaults to 256.
            max_dsp_path_length (int, optional): 15 for ddp/stanza(true_max_len=12), 10 for ltp(true_max_len=9). Defaults to 15.
            dsp_tool (str, optional): DSP tool used: ltp, ddp or stanza. Defaults to 'ddp'.
            tag2id (dict, optional): entity type to id dictionary. Defaults to None.
            use_attention4dsp (bool, optional): whether use attention for dsp path feature, otherwise use maxpool. Defaults to True.
            blank_padding (bool, optional): whether pad sequence to the same length. Defaults to True.
            mask_entity (bool, optional): whether do mask for entity. Defaults to False.
            compress_seq (bool, optional): whether compress sequence. Defaults to False.
            language (str, optional): language of bert model, support 'zh' or 'en'.

        Raises:
            NotImplementedError: DSP tool is not implemented.
        """
        super(XLNetEntityWithDSPEncoder, self).__init__(pretrain_path=pretrain_path, 
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
        xlnet_hidden_size = self.xlnet.config.hidden_size
        self.bilstm = nn.LSTM(input_size=xlnet_hidden_size, 
                            hidden_size=xlnet_hidden_size, 
                            num_layers=1, 
                            bidirectional=False, 
                            batch_first=True)
        self.compress_seq = compress_seq
        self.hidden_size = xlnet_hidden_size * 4
        # output map
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        # attention
        self.use_attention4dsp = use_attention4dsp
        # self.attention_map = nn.Linear(xlnet_hidden_size, xlnet_hidden_size)
        self.dsp_query = nn.Linear(xlnet_hidden_size, 1)

    def dsp_encode(self, hidden, dsp_path, dsp_path_length):
        # dsp_rep = torch.gather(hidden, 1, dsp_path.unsqueeze(-1).repeat(1, 1, hidden.size(-1)))
        # dsp_rep = torch.gather(hidden, 1, dsp_path.unsqueeze(-1).expand(*dsp_path.size(), hidden.size(-1)))
        dsp_rep = torch.stack([hidden[i, dsp_path[i]] for i in range(hidden.size(0))], dim=0) # (B, S, d)
        ## head entity dsp path representation
        if self.compress_seq:
            sorted_length_indices = dsp_path_length.argsort(descending=True) # (B,)
            unsorted_length_indices = sorted_length_indices.argsort(descending=False) # (B,)
            dsp_rep_packed = pack_padded_sequence(dsp_rep[sorted_length_indices], dsp_path_length[sorted_length_indices].detach().cpu(), batch_first=True)
            dsp_hidden_packed, _ = self.bilstm(dsp_rep_packed)
            dsp_hidden, _ = pad_packed_sequence(dsp_hidden_packed, batch_first=True) # (B, S, d)
            dsp_hidden = dsp_hidden[unsorted_length_indices] # (B, S, d), restore batch sequence
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

    def forward(self, seqs, pos1, pos2, token_type_ids, att_mask, ent_h_path, ent_t_path, ent_h_length, ent_t_length):
        """
        Args:
            seqs: (B, L), index of tokens
            pos1: (B, 1), position of the head entity starter
            pos2: (B, 1), position of the tail entity starter
            token_type_ids: (B, L), tokens type of xlnet
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Returns:
            (B, 2H), representations for sentences
        """
        hidden, _ = self.xlnet(seqs, token_type_ids=token_type_ids, attention_mask=att_mask)
        seq_len = torch.sum(att_mask, dim=-1)
        hidden_copy = hidden.clone()
        for i in range(seq_len.size(0)):
            hidden[i, :seq_len[i]] = hidden_copy[i, -seq_len[i]:]
            hidden[i, seq_len[i]:] = hidden_copy[i, :-seq_len[i]]
        hidden_copy = hidden_copy.detach().cpu()
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
        rep_out = torch.cat([head_hidden, tail_hidden, dsp_hidden], dim=-1)  # (B, 2d)
        # rep_out = torch.tanh(self.linear(rep_out)) # (B, 4d)
        rep_out = self.linear(rep_out)

        return rep_out

    def tokenize(self, item):
        """
        Args:
            item: data instance containing 'text' / 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        ret_items = super(XLNetEntityWithDSPEncoder, self).tokenize(item)
        if self.parser is not None:
            ent_h_pos_1 = torch.min(ret_items[1], ret_items[2]).item()
            ent_h_pos_2 = torch.min(ret_items[3], ret_items[4]).item()
            ent_t_pos_1 = torch.max(ret_items[1], ret_items[2]).item()
            ent_t_pos_2 = torch.max(ret_items[3], ret_items[4]).item()
            # shortest dependency path
            for i, t in enumerate(self.xlnet_tokens):
                if t.startswith('▁'):
                    self.xlnet_tokens[i] = t[1:]
            ent_h_path, ent_t_path = self.parser.parse(self.sentence, item['h'], item['t'], self.xlnet_tokens)
            ent_h_length = len(ent_h_path)
            ent_t_length = len(ent_t_path)
            if self.blank_padding:
                ent_h_path = ent_h_path[:self.max_dsp_path_length]
                ent_t_path = ent_t_path[:self.max_dsp_path_length]
            # move parsed token pos if entity type embedded
            for i, pos in enumerate(ent_h_path):
                if pos >= ent_h_pos_1:
                    pos += 1
                if pos >= ent_h_pos_2 - 1:
                    pos += 1
                if pos >= ent_t_pos_1:
                    pos += 1
                if pos >= ent_t_pos_2 - 1:
                    pos += 1
                ent_h_path[i] = pos
            for i, pos in enumerate(ent_t_path):
                if pos >= ent_h_pos_1:
                    pos += 1
                if pos >= ent_h_pos_2 - 1:
                    pos += 1
                if pos >= ent_t_pos_1:
                    pos += 1
                if pos >= ent_t_pos_2 - 1:
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



class XLNetEntityWithContextDSPEncoder(XLNetEntityWithDSPEncoder):
    def __init__(self, pretrain_path, max_length=256, max_dsp_path_length=15, dsp_tool='ddp', tag2id=None, 
                use_attention4dsp=True, use_attention4context=True, blank_padding=True, mask_entity=False, compress_seq=False, language='en'):
        """
        Args:
            pretrain_path (str): path of pretrain model
            max_length (int, optional): max_length of sequence. Defaults to 256.
            max_dsp_path_length (int, optional): 15 for ddp/stanza(true_max_len=12), 10 for ltp(true_max_len=9). Defaults to 15.
            dsp_tool (str, optional): DSP tool used: ltp, ddp or stanza. Defaults to 'ddp'.
            tag2id (dict, optional): entity type to id dictionary. Defaults to None.
            use_attention4dsp (bool, optional): whether use attention for dsp path feature, otherwise use maxpool. Defaults to True.
            use_attention4context (bool, optional): whether use attention for context. Defaults to True.
            blank_padding (bool, optional): whether pad sequence to the same length. Defaults to True.
            mask_entity (bool, optional): whether do mask for entity. Defaults to False.
            compress_seq (bool, optional): whether compress sequence. Defaults to False.
            language (str, optional): language of bert model, support 'zh' or 'en'.

        Raises:
            NotImplementedError: DSP tool is not implemented.
        """
        super(XLNetEntityWithContextDSPEncoder, self).__init__(pretrain_path=pretrain_path, 
                                                    max_length=max_length, 
                                                    max_dsp_path_length=max_dsp_path_length,
                                                    dsp_tool=dsp_tool,
                                                    tag2id=tag2id, 
                                                    use_attention4dsp=use_attention4dsp,
                                                    blank_padding=blank_padding, 
                                                    mask_entity=mask_entity,
                                                    compress_seq=compress_seq,
                                                    language=language)
        xlnet_hidden_size = self.xlnet.config.hidden_size
        self.hidden_size = xlnet_hidden_size * 5
        # output map
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        # attention
        self.use_attention4context = use_attention4context
        # self.attention_map = nn.Linear(xlnet_hidden_size, xlnet_hidden_size)
        if self.use_attention4context:
            self.context_query = nn.Linear(xlnet_hidden_size, 1)
        else:
            self.conv = nn.Conv1d(xlnet_hidden_size, xlnet_hidden_size, kernel_size=5)  # add a convolution layer to extract the global information of sentence

    def forward(self, seqs, pos1, pos2, token_type_ids, att_mask, ent_h_path, ent_t_path, ent_h_length, ent_t_length):
        """
        Args:
            seqs: (B, L), index of tokens
            pos1: (B, 1), position of the head entity starter
            pos2: (B, 1), position of the tail entity starter
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Returns:
            (B, 2H), representations for sentences
        """
        hidden, pooler_out = self.xlnet(seqs, token_type_ids=token_type_ids, attention_mask=att_mask)
        seq_len = torch.sum(att_mask, dim=-1)
        hidden_copy = hidden.clone()
        for i in range(seq_len.size(0)):
            hidden[i, :seq_len[i]] = hidden_copy[i, -seq_len[i]:]
            hidden[i, seq_len[i]:] = hidden_copy[i, :-seq_len[i]]
        hidden_copy = hidden_copy.detach().cpu()
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
            context_conv = self.conv(hidden.permute(0, 2, 1)) # (B, d, S)
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
        rep_out = torch.cat([head_hidden, tail_hidden, context_hidden, dsp_hidden], dim=-1)  # (B, 5d)
        # rep_out = torch.tanh(self.linear(rep_out)) # (B. 5d)
        rep_out = self.linear(rep_out)

        return rep_out
