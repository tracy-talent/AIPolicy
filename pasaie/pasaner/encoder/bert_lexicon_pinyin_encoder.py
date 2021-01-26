"""
 Author: liujian
 Date: 2021-01-19 23:38:22
 Last Modified by: liujian
 Last Modified time: 2021-01-19 23:38:22
"""

from ...tokenization.utils import convert_by_vocab
from ...module.nn.attention import dot_product_attention
from ...module.nn.cnn import masked_singlekernel_conv1d, masked_multikernel_conv1d

import logging
import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import BertModel, AlbertModel, BertTokenizer
from transformers.modeling_bert import BertEmbeddings


class BERT_Lexicon_PinYin_Word_Encoder(nn.Module):
    def __init__(self, 
                pretrain_path,
                word2id,
                word2pinyin,
                pinyin2id,
                word_size=50,
                lexicon_window_size=4,
                pinyin_size=50,
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
            lexicon_window_size (int, optional): upper bound(include) of lexicon match window size. Defaults to 4.
            max_length (int, optional): max length of sentence, used for postion embedding. Defaults to 512.
            max_pinyin_num_of_token (int, optional): max pinyin num of a token. Defaults to 10.
            bert_name (str): model name of bert series model, such as bert, roberta, xlnet, albert.
            blank_padding (bool, optional): whether pad sequence to max length. Defaults to True.
        """
        super(BERT_Lexicon_PinYin_Word_Encoder, self).__init__()

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

        
        self.word2id = word2id
        self.word2pinyin = word2pinyin
        self.pinyin2id = pinyin2id
        self.word_size = word_size
        self.lexicon_window_size = lexicon_window_size
        self.pinyin_size = pinyin_size
        self.hidden_size = self.bert.config.hidden_size
        self.max_length = max_length
        self.max_matched_lexcions = (1 + lexicon_window_size) * lexicon_window_size - 1
        self.max_pinyin_num_of_token = max_pinyin_num_of_token
        self.blank_padding = blank_padding
        # pinyin embedding matrix
        self.pinyin_embedding = nn.Embedding(len(self.pinyin2id), self.pinyin_size, padding_idx=self.pinyin2id['[PAD]'])
        # align word embedding and bert embedding
        self.lexicon2bert_linear = nn.Linear(self.word_size, self.hidden_size)
        self.pinyin2bert_linear = nn.Linear(self.pinyin_size, self.hidden_size)


    def forward(self, seqs_token_ids, seqs_lexicon_embed, seqs_pinyin_ids, att_pinyin_mask, att_lexicon_mask, att_token_mask):
        """
        Args:
            seqs: (B, L), index of tokens
            att_token_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        if 'roberta' in self.bert_name:
            # seq_out = self.bert(seqs, attention_mask=att_token_mask)[1][1] # hfl roberta
            bert_seq_embed, _ = self.bert(seqs_token_ids, attention_mask=att_token_mask) # clue-roberta
        else:
            bert_seq_embed, _ = self.bert(seqs_token_ids, attention_mask=att_token_mask)

        lexicon2bert_embed = self.lexicon2bert_linear(seqs_lexicon_embed)
        lexicon_att_output, _ = dot_product_attention(bert_seq_embed, lexicon2bert_embed, att_lexicon_mask)
        seqs_pinyin_embed = self.pinyin_embedding(seqs_pinyin_ids)
        pinyin2bert_embed = self.pinyin2bert_linear(seqs_pinyin_embed)
        pinyin_att_output, _ = self.dot_product_attention(bert_seq_embed, pinyin2bert_embed, att_pinyin_mask)
        inputs_embed = bert_seq_embed + lexicon_att_output + pinyin_att_output # (B, L, EMBED)
        return inputs_embed
    
    
    def lexicon_match(self, tokens):
        indexed_lexicons = [[self.word2id['[CLS]']] + [self.word2id['[PAD]']] * (self.max_matched_lexcions - 1)]
        for i in range(len(tokens)):
            words = []
            indexed_lexicons.append([])
            for w in range(self.lexicon_window_size, 1, -1):
                for p in range(w):
                    if i - p < 0:
                        break
                    if i - p + w > len(tokens):
                        continue
                    word = ''.join(tokens[i-p:i-p+w])
                    if word in self.word2id and word not in '～'.join(words):
                        words.append(word)
                        indexed_lexicons[-1].append(self.word2id[word])
            if len(indexed_lexicons[-1]) == 0:
                indexed_lexicons[-1].append(self.word2id['[UNK]'])
            indexed_lexicons[-1].extend([self.word2id['[PAD]']] * (self.max_matched_lexcions - len(indexed_lexicons[-1])))
        indexed_lexicons.append([self.word2id['[SEP]']] + [self.word2id['[PAD]']] * (self.max_matched_lexcions - 1))
        return indexed_lexicons


    def tokenize(self, *items): # items = (tokens, spans, [attrs, optional])
        """
        Args:
            items: (tokens, tags) or (tokens, spans, atrrs) or (sentence)
        Returns:
            indexed_tokens (torch.tensor): tokenizer encode ids of tokens, (1, L)
            att_token_mask (torch.tensor): token mask ids, (1, L)
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
        if self.blank_padding:
            is_truncated = False
            if len(indexed_tokens) <= self.max_length:
                bert_padding_idx = self.tokenizer.convert_tokens_to_ids('[PAD]')
                while len(indexed_tokens) < self.max_length:
                    indexed_tokens.append(bert_padding_idx)
                indexed_lexicons = self.lexicon_match(tokens[1:-1])
                for _ in range(self.max_length - len(tokens)):
                    indexed_lexicons.append([self.word2id['[PAD]']] * self.max_matched_lexcions)
            else:
                indexed_tokens[self.max_length - 1] = indexed_tokens[-1]
                indexed_tokens = indexed_tokens[:self.max_length]
                indexed_lexicons = self.lexicon_match(tokens[1:self.max_length - 1])
                is_truncated = True
            tokens_pinyinlist = convert_by_vocab(self.word2pinyin, tokens, max_seq_length=self.max_length, blank_id=[], unk_id=['[UNK]'])
            if is_truncated:
                tokens_pinyinlist[-1] = ['[UNK]']
        else:
            indexed_lexicons = self.lexicon_match(tokens[1:-1])
            tokens_pinyinlist = convert_by_vocab(self.word2pinyin, tokens, unk_id=[])    
            
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)
        indexed_lexicons = torch.tensor(indexed_lexicons).long().unsqueeze(0) # (1, L, W)
        indexed_token2pinyins = []
        for pinyinlist in tokens_pinyinlist:
            indexed_token2pinyins.append(convert_by_vocab(self.pinyin2id, pinyinlist, max_seq_length=self.max_pinyin_num_of_token, blank_id=self.pinyin2id['[PAD]'], unk_id=self.pinyin2id['[UNK]']))
        indexed_token2pinyins = torch.tensor(indexed_token2pinyins).long().unsqueeze(0) # (1, L, P)
        # attention mask
        att_token_mask = (indexed_tokens != self.tokenizer.convert_tokens_to_ids('[PAD]')).type(torch.uint8) # (1, L)
        att_lexicon_mask = (indexed_lexicons != self.word2id['[PAD]']).type(torch.uint8) # (1, L, W)
        att_pinyin_mask = (indexed_token2pinyins != self.pinyin2id['[PAD]']).type(torch.uint8) # (1, L, P)

        # ensure the first three is indexed_tokens and indexed_lexicons and indexed_token2pinyins, the last is att_token_mask
        return indexed_tokens, indexed_lexicons, indexed_token2pinyins, att_pinyin_mask, att_lexicon_mask, att_token_mask  


class BERT_Lexicon_PinYin_Char_Encoder(nn.Module):
    def __init__(self, 
                pretrain_path,
                word2id,
                word2pinyin,
                pinyin_char2id,
                word_size=50,
                lexicon_window_size=4,
                pinyin_char_size=50,
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
            lexicon_window_size (int, optional): upper bound(include) of lexicon match window size. Defaults to 4.
            max_length (int, optional): max length of sentence, used for postion embedding. Defaults to 512.
            max_pinyin_num_of_token (int, optional): max pinyin num of a token. Defaults to 10.
            max_pinyin_char_length (int, optional): max character length of a pinyin. Defaults to 7.
            bert_name (str): model name of bert series model, such as bert, roberta, xlnet, albert.
            blank_padding (bool, optional): whether pad sequence to max length. Defaults to True.
        """
        super(BERT_Lexicon_PinYin_Char_Encoder, self).__init__()

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

        self.word2id = word2id
        self.word2pinyin = word2pinyin
        self.pinyin_char2id = pinyin_char2id
        self.word_size = word_size
        self.lexicon_window_size = lexicon_window_size
        self.pinyin_char_size = pinyin_char_size
        self.hidden_size = self.bert.config.hidden_size
        self.max_length = max_length
        self.max_matched_lexcions = (1 + lexicon_window_size) * lexicon_window_size - 1
        self.max_pinyin_num_of_token = max_pinyin_num_of_token
        self.max_pinyin_char_length = max_pinyin_char_length
        self.blank_padding = blank_padding
        # pinyin character embedding matrix
        self.pinyin_char_embedding = nn.Embedding(len(self.pinyin_char2id), self.pinyin_char_size, padding_idx=self.pinyin_char2id['[PAD]'])
        # align word embedding and bert embedding
        self.lexicon2bert_linear = nn.Linear(self.word_size, self.hidden_size)
        # self.pinyin2bert_linear = nn.Linear(self.pinyin_char_size, self.hidden_size)
        self.char_conv = nn.Conv1d(self.pinyin_char_size, self.hidden_size, kernel_size=3, padding=1)
        self.masked_conv1d = masked_singlekernel_conv1d


    def forward(self, seqs_token_ids, seqs_lexicon_embed, seqs_pinyin_char_ids, att_pinyin_char_mask, att_lexicon_mask, att_token_mask):
        """
        Args:
            seqs: (B, L), index of tokens
            att_token_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        if 'roberta' in self.bert_name:
            # seq_out = self.bert(seqs, attention_mask=att_token_mask)[1][1] # hfl roberta
            bert_seq_embed, _ = self.bert(seqs_token_ids, attention_mask=att_token_mask) # clue-roberta
        else:
            bert_seq_embed, _ = self.bert(seqs_token_ids, attention_mask=att_token_mask)

        lexicon2bert_embed = self.lexicon2bert_linear(seqs_lexicon_embed)    
        lexicon_att_output, _ = dot_product_attention(bert_seq_embed, lexicon2bert_embed, att_lexicon_mask)
        seqs_pinyin_char_embed = self.pinyin_char_embedding(seqs_pinyin_char_ids)
        pinyin_conv = self.masked_conv1d(seqs_pinyin_char_embed, att_pinyin_char_mask, self.char_conv)
        # pinyin2bert_embed = self.pinyin2bert_linear(seqs_pinyin_embedding)
        pinyin_att_output, _ = dot_product_attention(bert_seq_embed, pinyin_conv, 
                                                    att_pinyin_char_mask.index_select(dim=-1, 
                                                    index=torch.tensor(0).to(att_pinyin_char_mask.device)).squeeze(-1) != self.pinyin_char2id['[PAD]'])
        inputs_embed = bert_seq_embed + lexicon_att_output + pinyin_att_output # (B, L, EMBED)
        return inputs_embed
    

    def lexicon_match(self, tokens):
        indexed_lexicons = [[self.word2id['[CLS]']] + [self.word2id['[PAD]']] * (self.max_matched_lexcions - 1)]
        for i in range(len(tokens)):
            words = []
            indexed_lexicons.append([])
            for w in range(self.lexicon_window_size, 1, -1):
                for p in range(w):
                    if i - p < 0:
                        break
                    if i - p + w > len(tokens):
                        continue
                    word = ''.join(tokens[i-p:i-p+w])
                    if word in self.word2id and word not in '～'.join(words):
                        words.append(word)
                        indexed_lexicons[-1].append(self.word2id[word])
            if len(indexed_lexicons[-1]) == 0:
                indexed_lexicons[-1].append(self.word2id['[UNK]'])
            indexed_lexicons[-1].extend([self.word2id['[PAD]']] * (self.max_matched_lexcions - len(indexed_lexicons[-1])))
        indexed_lexicons.append([self.word2id['[SEP]']] + [self.word2id['[PAD]']] * (self.max_matched_lexcions - 1))
        return indexed_lexicons


    def tokenize(self, *items): # items = (tokens, spans, [attrs, optional])
        """
        Args:
            items: (tokens, tags) or (tokens, spans, atrrs) or (sentence)
        Returns:
            indexed_tokens (torch.tensor): tokenizer encode ids of tokens, (1, L)
            att_token_mask (torch.tensor): token mask ids, (1, L)
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
        if self.blank_padding:
            is_truncated = False
            if len(indexed_tokens) <= self.max_length:
                bert_padding_idx = self.tokenizer.convert_tokens_to_ids('[PAD]')
                while len(indexed_tokens) < self.max_length:
                    indexed_tokens.append(bert_padding_idx)
                indexed_lexicons = self.lexicon_match(tokens[1:-1])
                for _ in range(self.max_length - len(tokens)):
                    indexed_lexicons.append([self.word2id['[PAD]']] * self.max_matched_lexcions)
            else:
                indexed_tokens[self.max_length - 1] = indexed_tokens[-1]
                indexed_tokens = indexed_tokens[:self.max_length]
                indexed_lexicons = self.lexicon_match(tokens[1:self.max_length - 1])
                is_truncated = True
            tokens_pinyinlist = convert_by_vocab(self.word2pinyin, tokens, max_seq_length=self.max_length, blank_id=[], unk_id=[])
            if is_truncated:
                tokens_pinyinlist[-1] = []
        else:
            indexed_lexicons = self.lexicon_match(tokens[1:-1])
            tokens_pinyinlist = convert_by_vocab(self.word2pinyin, tokens, unk_id=[])    
            
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)
        indexed_lexicons = torch.tensor(indexed_lexicons).long().unsqueeze(0) # (1, L, W)
        indexed_token2pinyins_chars = []
        for pinyinlist in tokens_pinyinlist:
            indexed_token2pinyins_chars.append([])
            for pinyin in pinyinlist:
                indexed_token2pinyins_chars[-1].append(convert_by_vocab(self.pinyin_char2id, list(pinyin), max_seq_length=self.max_pinyin_char_length, blank_id=self.pinyin_char2id['[PAD]'], unk_id=self.pinyin_char2id['[UNK]']))
            for _ in range(self.max_pinyin_num_of_token - len(pinyinlist)):
                indexed_token2pinyins_chars[-1].append([self.pinyin_char2id['[PAD]']] * self.max_pinyin_char_length)
        indexed_token2pinyins_chars = torch.tensor(indexed_token2pinyins_chars).unsqueeze(0) # (1, L, P)
        # attention mask
        att_token_mask = (indexed_tokens != self.tokenizer.convert_tokens_to_ids('[PAD]')).type(torch.uint8) # (1, L)
        att_lexicon_mask = (indexed_lexicons != self.word2id['[PAD]']).type(torch.uint8) # (1, L, W)
        att_pinyin_char_mask = (indexed_token2pinyins_chars != self.pinyin_char2id['[PAD]']).type(torch.uint8) # (1, L, P)

        # ensure the first two is indexed_tokens and indexed_token2word, the last is att_token_mask
        return indexed_tokens, indexed_lexicons, indexed_token2pinyins_chars, att_pinyin_char_mask, att_lexicon_mask, att_token_mask  



class BERT_Lexicon_PinYin_Char_MultiConv_Encoder(BERT_Lexicon_PinYin_Char_Encoder):
    def __init__(self, 
                pretrain_path,
                word2id,
                word2pinyin,
                pinyin_char2id,
                word_size=50,
                lexicon_window_size=4,
                pinyin_char_size=50,
                max_length=512, 
                max_pinyin_num_of_token=10,
                max_pinyin_char_length=7,
                bert_name='bert', 
                blank_padding=True,
                convs_config=[(256, 2), (256, 3), (256, 4)]):
        super(BERT_Lexicon_PinYin_Char_MultiConv_Encoder, self).__init__(
            pretrain_path=pretrain_path,
            word2id=word2id,
            word2pinyin=word2pinyin,
            pinyin_char2id=pinyin_char2id,
            word_size=word_size,
            lexicon_window_size=lexicon_window_size,
            pinyin_char_size=pinyin_char_size,
            max_length=max_length, 
            max_pinyin_num_of_token=max_pinyin_num_of_token,
            max_pinyin_char_length=max_pinyin_char_length,
            bert_name=bert_name, 
            blank_padding=blank_padding
        )
        assert self.hidden_size == sum(cc[0] for cc in convs_config)
        self.char_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=self.pinyin_char_size, out_channels=oc, kernel_size=ks),
                nn.MaxPool1d(kernel_size=self.max_pinyin_char_length - ks + 1),
                nn.ReLU()
            )
            for oc, ks in convs_config
        ])
        self.masked_conv1d = masked_multikernel_conv1d
