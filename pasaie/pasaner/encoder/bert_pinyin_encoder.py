"""
 Author: liujian
 Date: 2021-01-21 22:44:00
 Last Modified by: liujian
 Last Modified time: 2021-01-21 22:44:00
"""

from ...tokenization import JiebaTokenizer
from ...tokenization.utils import convert_by_vocab

import logging
import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import BertModel, AlbertModel, BertTokenizer
from transformers.modeling_bert import BertEmbeddings


class BERT_PinYin_Word_Encoder(nn.Module):
    def __init__(self, 
                pretrain_path,
                word2pinyin,
                pinyin2id,
                pinyin_size=50,
                custom_dict=None,
                max_length=512, 
                max_pinyin_num_of_token=10,
                bert_name='bert', 
                blank_padding=True):
        """
        Args:
            pretrain_path (str): path of pretrain model.
            word2id (dict): dictionary of word->idx mapping.
            word2pinyin (dict): dictionary of word -> [pinyins] mapping.
            pinyin2id (dict): dictionary of pinyin->idx mapping.
            word_size (int, optional): size of word embedding. Defaults to 50.
            custom_dict (str, optional): customized dictionary for word tokenizer. Defaults to None.
            max_length (int, optional): max length of sentence, used for postion embedding. Defaults to 512.
            max_pinyin_num_of_token (int, optional): max pinyin num of a token. Defaults to 10.
            bert_name (str): model name of bert series model, such as bert, roberta, xlnet, albert.
            blank_padding (bool, optional): whether pad sequence to max length. Defaults to True.
        """
        super(BERT_PinYin_Word_Encoder, self).__init__()

        # load bert model and bert tokenizer
        logging.info(f'Loading {bert_name} pre-trained checkpoint.')
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
        # add missed tokens in vocab.txt
        num_added_tokens = self.tokenizer.add_tokens(['“', '”', '—'])
        print(f"we have added {num_added_tokens} tokens ['“', '”', '—']")
        self.bert.resize_token_embeddings(len(self.tokenizer))
        # self.embeddings = BertEmbeddings(self.bert.config)

        
        self.word2pinyin = word2pinyin
        self.pinyin2id = pinyin2id
        self.pinyin_size = pinyin_size
        # self.hidden_size = self.bert.config.hidden_size + self.word_size
        self.hidden_size = self.bert.config.hidden_size
        self.max_length = max_length
        self.max_pinyin_num_of_token = max_pinyin_num_of_token
        self.blank_padding = blank_padding
        # pinyin embedding matrix
        self.pinyin_embedding = nn.Embedding(len(self.pinyin2id), self.pinyin_size, padding_idx=self.pinyin2id['[PAD]'])
        # align word embedding and bert embedding
        self.pinyin2bert_linear = nn.Linear(self.pinyin_size, self.hidden_size)


    def dot_product_attention(self, att_query, att_kv, att_mask):
        att_score = torch.matmul(att_kv, att_query.unsqueeze(-1)).squeeze(-1)
        att_score[att_mask == 0] = 1e-9
        att_weight = F.softmax(att_score, dim=-1)
        att_output = torch.matmul(att_weight.unsqueeze(-2), att_kv).squeeze(-2)
        return att_output, att_weight.data


    def forward(self, seqs_char, seqs_pinyin_ids, att_pinyin_mask, att_mask):
        """
        Args:
            seqs: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        if 'roberta' in self.bert_name:
            # seq_out = self.bert(seqs, attention_mask=att_mask)[1][1] # hfl roberta
            bert_seq_embed, _ = self.bert(seqs_char, attention_mask=att_mask) # clue-roberta
        else:
            bert_seq_embed, _ = self.bert(seqs_char, attention_mask=att_mask)
        # seq_embedding = self.embeddings(seqs)
        # inputs_embed = torch.cat([
        #     bert_seq_embed,
        #     self.word2bert_linear(seqs_word_embedding)
        # ], dim=-1) # (B, L, EMBED)
        seqs_pinyin_embed = self.pinyin_embedding(seqs_pinyin_ids)
        pinyin2bert_embed = self.pinyin2bert_linear(seqs_pinyin_embed)
        pinyin_att_output, _ = self.dot_product_attention(bert_seq_embed, pinyin2bert_embed, att_pinyin_mask)
        inputs_embed = bert_seq_embed + pinyin_att_output # (B, L, EMBED)
        return inputs_embed
    

    def tokenize(self, *items): # items = (tokens, spans, [attrs, optional])
        """
        Args:
            items: (tokens, tags) or (tokens, spans, atrrs) or (sentence)
        Returns:
            indexed_tokens (torch.tensor): tokenizer encode ids of tokens, (1, L)
            att_mask (torch.tensor): token mask ids, (1, L)
        """
        if isinstance(items[0], str):
            sentence = items[0]
            is_token = False
        else:
            sentence = items[0]
            is_token = True
        if is_token:
            items[0].insert(0, '[CLS]')
            items[0].append('[SEP]')
            if len(items) > 1:
                items[1].insert(0, 'O')
                items[1].append('O')
            if len(items) > 2:
                items[2].insert(0, 'null')
                items[2].append('null')
            tokens = items[0]
        else:
            tokens = ['[CLS]'] + self.tokenizer.tokenize(sentence) + ['[SEP]']

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        avail_len = torch.tensor([len(indexed_tokens)])

        if self.blank_padding:
            is_truncated = False
            if len(indexed_tokens) <= self.max_length:
                while len(indexed_tokens) < self.max_length:
                    indexed_tokens.append(0)
            else:
                indexed_tokens[self.max_length - 1] = indexed_tokens[-1]
                indexed_tokens = indexed_tokens[:self.max_length]
                is_truncated = True
            tokens_pinyinlist = convert_by_vocab(self.word2pinyin, tokens, max_seq_length=self.max_length, blank_id=[], unk_id=['[UNK]'])
            if is_truncated:
                tokens_pinyinlist[-1] = ['[UNK]']
        else:
            tokens_pinyinlist = convert_by_vocab(self.word2pinyin, tokens, unk_id=[])    
            
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)
        indexed_token2pinyins = []
        for pinyinlist in tokens_pinyinlist:
            indexed_token2pinyins.append(convert_by_vocab(self.pinyin2id, pinyinlist, max_seq_length=self.max_pinyin_num_of_token, blank_id=self.pinyin2id['[PAD]'], unk_id=self.pinyin2id['[UNK]']))
        indexed_token2pinyins = torch.tensor(indexed_token2pinyins).long().unsqueeze(0)
        # attention mask
        att_mask = torch.zeros(indexed_tokens.size(), dtype=torch.uint8) # (1, L)
        att_mask[0, :avail_len] = 1
        att_pinyin_mask = (indexed_token2pinyins != self.pinyin2id['[PAD]']).type(torch.uint8)

        # ensure the first is indexed_tokens, the last is att_mask
        return indexed_tokens, indexed_token2pinyins, att_pinyin_mask, att_mask  


class BERT_PinYin_Char_Encoder(nn.Module):
    def __init__(self, 
                pretrain_path,
                word2pinyin,
                pinyin_char2id,
                pinyin_char_size=50,
                custom_dict=None,
                max_length=512, 
                max_pinyin_num_of_token=10,
                max_pinyin_char_length=7,
                bert_name='bert', 
                blank_padding=True):
        """
        Args:
            pretrain_path (str): path of pretrain model.
            word2id (dict): dictionary of word -> idx mapping.
            word2pinyin (dict): dictionary of word -> [pinyins] mapping.
            pinyin_char2id (dict): dictionary of pinyin character -> idx mapping.
            word_size (int, optional): size of word embedding. Defaults to 50.
            custom_dict (str, optional): customized dictionary for word tokenizer. Defaults to None.
            max_length (int, optional): max length of sentence, used for postion embedding. Defaults to 512.
            max_pinyin_num_of_token (int, optional): max pinyin num of a token. Defaults to 10.
            max_pinyin_char_length (int, optional): max character length of a pinyin. Defaults to 7.
            bert_name (str): model name of bert series model, such as bert, roberta, xlnet, albert.
            blank_padding (bool, optional): whether pad sequence to max length. Defaults to True.
        """
        super(BERT_PinYin_Char_Encoder, self).__init__()

        # load bert model and bert tokenizer
        logging.info(f'Loading {bert_name} pre-trained checkpoint.')
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
        # add missed tokens in vocab.txt
        num_added_tokens = self.tokenizer.add_tokens(['“', '”', '—'])
        print(f"we have added {num_added_tokens} tokens ['“', '”', '—']")
        self.bert.resize_token_embeddings(len(self.tokenizer))
        # self.embeddings = BertEmbeddings(self.bert.config)

        self.word2pinyin = word2pinyin
        self.pinyin_char2id = pinyin_char2id
        self.pinyin_char_size = pinyin_char_size
        self.hidden_size = self.bert.config.hidden_size
        self.max_length = max_length
        self.max_pinyin_num_of_token = max_pinyin_num_of_token
        self.max_pinyin_char_length = max_pinyin_char_length
        self.blank_padding = blank_padding
        # pinyin character embedding matrix
        self.pinyin_char_embedding = nn.Embedding(len(self.pinyin_char2id), self.pinyin_char_size, padding_idx=self.pinyin_char2id['[PAD]'])
        # self.pinyin2bert_linear = nn.Linear(self.pinyin_char_size, self.hidden_size)
        self.char_conv = nn.Conv1d(self.pinyin_char_size, self.hidden_size, kernel_size=3, padding=1)


    def masked_conv1d(self, hiddens, weights):
        shape = hiddens.size()
        dim1 = functools.reduce(lambda x, y: x * y, shape[:-2])
        dim2 = shape[-2]
        dim3 = shape[-1]
        hiddens = hiddens.contiguous().resize(dim1, dim2, dim3).transpose(-2, -1)
        weights = weights.contiguous().resize(dim1, dim2, 1).float().transpose(-2, -1)
        hiddens *= weights
        conv_hiddens = self.char_conv(hiddens)
        conv_hiddens *= weights
        conv_hiddens = F.relu(F.max_pool1d(conv_hiddens, conv_hiddens.size(-1)).squeeze(-1))
        conv_hiddens = conv_hiddens.contiguous().resize(*(shape[:-2] + conv_hiddens.size()[-1:]))
        return conv_hiddens


    def dot_product_attention(self, att_query, att_kv, att_mask):
        att_score = torch.matmul(att_kv, att_query.unsqueeze(-1)).squeeze(-1)
        att_score[att_mask == 0] = 1e-9
        att_weight = F.softmax(att_score, dim=-1)
        att_output = torch.matmul(att_weight.unsqueeze(-2), att_kv).squeeze(-2)
        return att_output, att_weight.data


    def forward(self, seqs_char, seqs_pinyin_char_ids, att_pinyin_char_mask, att_mask):
        """
        Args:
            seqs: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        if 'roberta' in self.bert_name:
            # seq_out = self.bert(seqs, attention_mask=att_mask)[1][1] # hfl roberta
            bert_seq_embed, _ = self.bert(seqs_char, attention_mask=att_mask) # clue-roberta
        else:
            bert_seq_embed, _ = self.bert(seqs_char, attention_mask=att_mask)
        
        seqs_pinyin_char_embed = self.pinyin_char_embedding(seqs_pinyin_char_ids)
        pinyin_conv = self.masked_conv1d(seqs_pinyin_char_embed, att_pinyin_char_mask)
        # pinyin2bert_embed = self.pinyin2bert_linear(seqs_pinyin_embedding)
        pinyin_att_output, _ = self.dot_product_attention(bert_seq_embed, pinyin_conv, 
                                                    att_pinyin_char_mask.index_select(dim=-1, 
                                                    index=torch.tensor(0).to(att_pinyin_char_mask.device)).squeeze(-1) != self.pinyin_char2id['[PAD]'])
        inputs_embed = bert_seq_embed + pinyin_att_output # (B, L, EMBED)
        return inputs_embed
    

    def tokenize(self, *items): # items = (tokens, spans, [attrs, optional])
        """
        Args:
            items: (tokens, tags) or (tokens, spans, atrrs) or (sentence)
        Returns:
            indexed_tokens (torch.tensor): tokenizer encode ids of tokens, (1, L)
            att_mask (torch.tensor): token mask ids, (1, L)
        """
        if isinstance(items[0], str):
            sentence = items[0]
            is_token = False
        else:
            sentence = items[0]
            is_token = True
        if is_token:
            items[0].insert(0, '[CLS]')
            items[0].append('[SEP]')
            if len(items) > 1:
                items[1].insert(0, 'O')
                items[1].append('O')
            if len(items) > 2:
                items[2].insert(0, 'null')
                items[2].append('null')
            tokens = items[0]
        else:
            tokens = ['[CLS]'] + self.tokenizer.tokenize(sentence) + ['[SEP]']

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        avail_len = torch.tensor([len(indexed_tokens)])

        if self.blank_padding:
            is_truncated = False
            if len(indexed_tokens) <= self.max_length:
                while len(indexed_tokens) < self.max_length:
                    indexed_tokens.append(0)
            else:
                indexed_tokens[self.max_length - 1] = indexed_tokens[-1]
                indexed_tokens = indexed_tokens[:self.max_length]
                is_truncated = True
            tokens_pinyinlist = convert_by_vocab(self.word2pinyin, tokens, max_seq_length=self.max_length, blank_id=[], unk_id=[])
            if is_truncated:
                tokens_pinyinlist[-1] = []
        else:
            tokens_pinyinlist = convert_by_vocab(self.word2pinyin, tokens, unk_id=[])    
            
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)
        indexed_token2pinyins_chars = []
        for pinyinlist in tokens_pinyinlist:
            indexed_token2pinyins_chars.append([])
            for pinyin in pinyinlist:
                indexed_token2pinyins_chars[-1].append(convert_by_vocab(self.pinyin_char2id, list(pinyin), max_seq_length=self.max_pinyin_char_length, blank_id=self.pinyin_char2id['[PAD]'], unk_id=self.pinyin_char2id['[UNK]']))
            for _ in range(self.max_pinyin_num_of_token - len(pinyinlist)):
                indexed_token2pinyins_chars[-1].append([self.pinyin_char2id['[PAD]']] * self.max_pinyin_char_length)
        indexed_token2pinyins_chars = torch.tensor(indexed_token2pinyins_chars).unsqueeze(0)
        # attention mask
        att_mask = torch.zeros(indexed_tokens.size(), dtype=torch.uint8) # (1, L)
        att_mask[0, :avail_len] = 1
        att_pinyin_char_mask = (indexed_token2pinyins_chars != self.pinyin_char2id['[PAD]']).type(torch.uint8)

        # ensure the first is indexed_tokens, the last is att_mask
        return indexed_tokens, indexed_token2pinyins_chars, att_pinyin_char_mask, att_mask  



class BERT_PinYin_Char_MultiConv_Encoder(BERT_PinYin_Char_Encoder):
    def __init__(self, 
                pretrain_path,
                word2pinyin,
                pinyin_char2id,
                pinyin_char_size=50,
                custom_dict=None,
                max_length=512, 
                max_pinyin_num_of_token=10,
                max_pinyin_char_length=7,
                bert_name='bert', 
                blank_padding=True,
                convs_config=[(256, 2), (256, 3), (256, 4)]):
        super(BERT_PinYin_Char_MultiConv_Encoder, self).__init__(
            pretrain_path=pretrain_path,
            word2pinyin=word2pinyin,
            pinyin_char2id=pinyin_char2id,
            pinyin_char_size=pinyin_char_size,
            custom_dict=custom_dict,
            max_length=max_length, 
            max_pinyin_num_of_token=max_pinyin_num_of_token,
            max_pinyin_char_length=max_pinyin_char_length,
            bert_name=bert_name, 
            blank_padding=blank_padding
        )
        del self.char_conv
        assert self.hidden_size == sum(cc[0] for cc in convs_config)
        self.char_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=self.pinyin_char_size, out_channels=oc, kernel_size=ks),
                nn.MaxPool1d(kernel_size=self.max_pinyin_char_length - ks + 1),
                nn.ReLU()
            )
            for oc, ks in convs_config
        ])
    

    def masked_conv1d(self, hiddens, weights):
        shape = hiddens.size()
        dim1 = functools.reduce(lambda x, y: x * y, shape[:-2])
        dim2 = shape[-2]
        dim3 = shape[-1]
        hiddens = hiddens.contiguous().resize(dim1, dim2, dim3).transpose(-2, -1)
        weights = weights.contiguous().resize(dim1, dim2, 1).float().transpose(-2, -1)
        hiddens *= weights
        convs_out = [conv(hiddens).squeeze(-1) for conv in self.char_convs]
        conv_hiddens = torch.cat(convs_out, dim=-1)
        conv_hiddens = conv_hiddens.contiguous().resize(*(shape[:-2] + conv_hiddens.size()[-1:]))
        return conv_hiddens
