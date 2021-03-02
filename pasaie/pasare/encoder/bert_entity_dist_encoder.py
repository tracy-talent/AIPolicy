"""
 Author: liujian
 Date: 2021-02-28 15:57:33
 Last Modified by: liujian
 Last Modified time: 2021-02-28 15:57:33
"""

from .bert_encoder import BERTEntityEncoder, BERTEntityWithDSPEncoder
from ...module.pool import PieceMaxPool

import torch
from torch import nn


class BERTEntityDistEncoder(BERTEntityEncoder):
    def __init__(self, pretrain_path, bert_name='bert', max_length=256, position_size=5, tag2id=None, 
                blank_padding=True, mask_entity=False, language='en'):
        super(BERTEntityDistEncoder, self).__init__(pretrain_path=pretrain_path, 
                                                    bert_name=bert_name,
                                                    max_length=max_length, 
                                                    tag2id=tag2id, 
                                                    blank_padding=blank_padding, 
                                                    mask_entity=mask_entity,
                                                    language=language)
        self.position_size = position_size
        self.pos1_embedding = nn.Embedding(2 * max_length, self.position_size, padding_idx=0)
        self.pos2_embedding = nn.Embedding(2 * max_length, self.position_size, padding_idx=0)
        emb_size = self.bert.config.hidden_size + position_size * 2
        self.hidden_size = emb_size * 2
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, seqs, pos1, pos2, att_mask):
        ent1_dist = []
        ent2_dist = []
        seq_size = seqs.size(1)
        for i in range(att_mask.size(0)):
            seq_len = (att_mask[i] != 0).sum()
            ent_pos_1 = pos1[i].item()
            ent1_dist.append([min(j - ent_pos_1 + self.max_length, 2 * self.max_length - 1) for j in range(seq_len)])
            ent1_dist[-1].extend([0] * (seq_size - seq_len))
            ent_pos_2 = pos2[i].item()
            ent2_dist.append([min(j - ent_pos_2 + self.max_length, 2 * self.max_length - 1) for j in range(seq_len)])
            ent2_dist[-1].extend([0] * (seq_size - seq_len))
        ent1_dist = torch.tensor(ent1_dist).to(seqs.device)
        ent2_dist = torch.tensor(ent2_dist).to(seqs.device)

        hidden, _ = self.bert(seqs, attention_mask=(att_mask != 0).byte())
        hidden = torch.cat([hidden, 
                       self.pos1_embedding(ent1_dist), 
                       self.pos2_embedding(ent2_dist)], 2) # (B, L, H)
        # Get entity start hidden state
        onehot_head = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_tail = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_head = onehot_head.scatter_(1, pos1, 1)
        onehot_tail = onehot_tail.scatter_(1, pos2, 1)
        head_hidden = torch.matmul(onehot_head.unsqueeze(1), hidden).squeeze(1)  # (B, H)
        tail_hidden = torch.matmul(onehot_tail.unsqueeze(1), hidden).squeeze(1)  # (B, H)

        rep_out = torch.cat([head_hidden, tail_hidden], dim=-1)
        rep_out = self.linear(rep_out)

        return rep_out



class BERTEntityDistWithPCNNEncoder(BERTEntityEncoder):
    def __init__(self, pretrain_path, bert_name='bert', max_length=256, position_size=5, tag2id=None, 
                blank_padding=True, mask_entity=False, language='en'):
        super(BERTEntityDistWithPCNNEncoder, self).__init__(pretrain_path=pretrain_path, 
                                                    bert_name=bert_name,
                                                    max_length=max_length, 
                                                    tag2id=tag2id, 
                                                    blank_padding=blank_padding, 
                                                    mask_entity=mask_entity,
                                                    language=language)
        self.position_size = position_size
        self.pos1_embedding = nn.Embedding(2 * max_length, self.position_size, padding_idx=0)
        self.pos2_embedding = nn.Embedding(2 * max_length, self.position_size, padding_idx=0)                       
        emb_size = self.bert.config.hidden_size + position_size * 2
        assert emb_size // 3 == 0
        self.hidden_size = emb_size * 3
        self.conv = nn.Conv1d(emb_size, emb_size // 3, 3)
        self.pool = PieceMaxPool(piece_num=3)
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)


    def forward(self, seqs, pos1, pos2, att_mask):
        ent1_dist = []
        ent2_dist = []
        seq_size = seqs.size(1)
        seq_len = (att_mask != 0).sum(dim=-1)
        for i in range(att_mask.size(0)):
            ent_pos_1 = pos1[i].item()
            ent1_dist.append([min(j - ent_pos_1 + self.max_length, 2 * self.max_length - 1) for j in range(seq_len[i])])
            ent1_dist[-1].extend([0] * (seq_size - seq_len[i]))
            ent_pos_2 = pos2[i].item()
            ent2_dist.append([min(j - ent_pos_2 + self.max_length, 2 * self.max_length - 1) for j in range(seq_len[i])])
            ent2_dist[-1].extend([0] * (seq_size - seq_len[i]))
        ent1_dist = torch.tensor(ent1_dist).to(seqs.device)
        ent2_dist = torch.tensor(ent2_dist).to(seqs.device)

        hidden, _ = self.bert(seqs, attention_mask=(att_mask != 0).byte())
        hidden = torch.cat([hidden, 
                       self.pos1_embedding(ent1_dist), 
                       self.pos2_embedding(ent2_dist)], dim=-1) # (B, L, H)
        # Get entity start hidden state
        onehot_head = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_tail = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_head = onehot_head.scatter_(1, pos1, 1)
        onehot_tail = onehot_tail.scatter_(1, pos2, 1)
        head_hidden = torch.matmul(onehot_head.unsqueeze(1), hidden).squeeze(1)  # (B, H)
        tail_hidden = torch.matmul(onehot_tail.unsqueeze(1), hidden).squeeze(1)  # (B, H)

        # get pcnn hidden
        hidden = hidden.transpose(1, 2) # (B, H, L)
        pcnn_hidden = torch.relu(self.conv(hidden)) # (B, H, L)
        pcnn_hidden = self.pool(hidden, att_mask)

        rep_out = torch.cat([head_hidden, tail_hidden, pcnn_hidden], dim=-1)
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
        ret_items = super(BERTEntityDistWithPCNNEncoder, self).tokenize(item)
        pos_min = min(ret_items[3], ret_items[4])
        pos_max = max(ret_items[3], ret_items[4])
        ret_items[-1][0, :pos_min] = 1
        ret_items[-1][0, pos_min:pos_max] = 2
        ret_items[-1][0, pos_max:] = 3

        return ret_items



class BERTEntityDistWithDSPEncoder(BERTEntityWithDSPEncoder):
    def __init__(self, pretrain_path, bert_name='bert', max_length=256, max_dsp_path_length=15, position_size=5, dsp_tool='ddp', 
                tag2id=None, use_attention4dsp=True, blank_padding=True, mask_entity=False, compress_seq=False, language='en'):
        super(BERTEntityDistWithDSPEncoder, self).__init__(pretrain_path=pretrain_path, 
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
        self.position_size = position_size
        self.pos1_embedding = nn.Embedding(2 * max_length, self.position_size, padding_idx=0)
        self.pos2_embedding = nn.Embedding(2 * max_length, self.position_size, padding_idx=0)
        emb_size = self.bert.config.hidden_size + position_size * 2
        self.hidden_size = emb_size * 4
        self.bilstm = nn.LSTM(input_size=emb_size, 
                            hidden_size=emb_size, 
                            num_layers=1, 
                            bidirectional=False, 
                            batch_first=True)
        self.dsp_query = nn.Linear(emb_size, 1)
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
    

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
        ent1_dist = []
        ent2_dist = []
        seq_size = seqs.size(1)
        seq_len = (att_mask != 0).sum(dim=-1)
        for i in range(att_mask.size(0)):
            ent_pos_1 = pos1[i].item()
            ent1_dist.append([min(j - ent_pos_1 + self.max_length, 2 * self.max_length - 1) for j in range(seq_len[i])])
            ent1_dist[-1].extend([0] * (seq_size - seq_len[i]))
            ent_pos_2 = pos2[i].item()
            ent2_dist.append([min(j - ent_pos_2 + self.max_length, 2 * self.max_length - 1) for j in range(seq_len[i])])
            ent2_dist[-1].extend([0] * (seq_size - seq_len[i]))
        ent1_dist = torch.tensor(ent1_dist).to(seqs.device)
        ent2_dist = torch.tensor(ent2_dist).to(seqs.device)

        hidden, _ = self.bert(seqs, attention_mask=(att_mask != 0).byte())
        hidden = torch.cat([hidden, 
                       self.pos1_embedding(ent1_dist), 
                       self.pos2_embedding(ent2_dist)], 2) # (B, L, H)
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
        dsp_hidden = torch.cat([ent_h_dsp_hidden, ent_t_dsp_hidden], dim=-1) # (B, 2d)

        # gather all features
        rep_out = torch.cat([head_hidden, tail_hidden, dsp_hidden], dim=-1)  # (B, 4H)
        # rep_out = torch.tanh(self.linear(rep_out))
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
        ret_items = super(BERTEntityDistWithDSPEncoder, self).tokenize(item)
        pos_min = min(ret_items[3], ret_items[4])
        pos_max = max(ret_items[3], ret_items[4])
        ret_items[5][0, :pos_min] = 1
        ret_items[5][0, pos_min:pos_max] = 2
        ret_items[5][0, pos_max:] = 3

        return ret_items



class BERTEntityDistWithPCNNDSPEncoder(BERTEntityDistWithDSPEncoder):
    def __init__(self, pretrain_path, bert_name='bert', max_length=256, max_dsp_path_length=15, position_size=5, dsp_tool='ddp', 
                tag2id=None, use_attention4dsp=True, blank_padding=True, mask_entity=False, compress_seq=False, language='en'):
        super(BERTEntityDistWithPCNNDSPEncoder, self).__init__(pretrain_path=pretrain_path, 
                                                    bert_name=bert_name, 
                                                    max_length=max_length, 
                                                    max_dsp_path_length=max_dsp_path_length, 
                                                    position_size=position_size, 
                                                    dsp_tool=dsp_tool, 
                                                    tag2id=tag2id, 
                                                    use_attention4dsp=use_attention4dsp, 
                                                    blank_padding=blank_padding, 
                                                    mask_entity=mask_entity,
                                                    compress_seq=compress_seq, 
                                                    language=language)
        emb_size = self.bert.config.hidden_size + position_size * 2
        assert emb_size // 3 == 0
        self.hidden_size = emb_size * 5
        self.conv = nn.Conv1d(emb_size, emb_size // 3, kernel_size=3)
        self.pool = PieceMaxPool(piece_num=3)
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
    

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
        ent1_dist = []
        ent2_dist = []
        seq_size = seqs.size(1)
        seq_len = (att_mask != 0).sum(dim=-1)
        for i in range(att_mask.size(0)):
            ent_pos_1 = pos1[i].item()
            ent1_dist.append([min(j - ent_pos_1 + self.max_length, 2 * self.max_length - 1) for j in range(seq_len[i])])
            ent1_dist[-1].extend([0] * (seq_size - seq_len[i]))
            ent_pos_2 = pos2[i].item()
            ent2_dist.append([min(j - ent_pos_2 + self.max_length, 2 * self.max_length - 1) for j in range(seq_len[i])])
            ent2_dist[-1].extend([0] * (seq_size - seq_len[i]))
        ent1_dist = torch.tensor(ent1_dist).to(seqs.device)
        ent2_dist = torch.tensor(ent2_dist).to(seqs.device)

        hidden, _ = self.bert(seqs, attention_mask=(att_mask != 0).byte())
        hidden = torch.cat([hidden, 
                       self.pos1_embedding(ent1_dist), 
                       self.pos2_embedding(ent2_dist)], 2) # (B, L, H)
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
        dsp_hidden = torch.cat([ent_h_dsp_hidden, ent_t_dsp_hidden], dim=-1) # (B, 2d)

        # get pcnn hidden
        hidden = hidden.transpose(1, 2) # (B, H, L)
        pcnn_hidden = torch.relu(self.conv(hidden)) # (B, H, L)
        pcnn_hidden = self.pool(hidden, att_mask)

        # gather all features
        rep_out = torch.cat([head_hidden, tail_hidden, pcnn_hidden, dsp_hidden], dim=-1)  # (B, 4H)
        # rep_out = torch.tanh(self.linear(rep_out))
        rep_out = self.linear(rep_out)

        return rep_out
