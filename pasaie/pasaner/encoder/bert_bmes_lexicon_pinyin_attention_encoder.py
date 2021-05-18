"""
 Author: liujian
 Date: 2021-01-19 23:38:22
 Last Modified by: liujian
 Last Modified time: 2021-01-19 23:38:22
"""

from ...tokenization.utils import strip_accents
from ...utils.common import is_eng_word, is_digit, is_pinyin
from ...module.nn.attention import dot_product_attention, dot_product_attention_with_project
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
from pypinyin import lazy_pinyin, Style


class BERT_BMES_Lexicon_PinYin_Word_Attention_Cat_Encoder(nn.Module):
    def __init__(self, 
                pretrain_path,
                word2id,
                pinyin2id,
                pinyin_embedding=None,
                word_size=50,
                lexicon_window_size=4,
                pinyin_size=50,
                max_length=512, 
                group_num=3,
                bert_name='bert', 
                blank_padding=True):
        """
        Args:
            pretrain_path (str): path of pretrain model.
            word2id (dict): dictionary of word->idx mapping.
            pinyin2id (dict): dictionary of pinyin->idx mapping.
            pinyin_embedding (nn.Embedding): pinyin embedding. Defaults to None.
            word_size (int, optional): size of word embedding. Defaults to 50.
            lexicon_window_size (int, optional): upper bound(include) of lexicon match window size. Defaults to 4.
            max_length (int, optional): max length of sentence, used for postion embedding. Defaults to 512.
            group_num (int, optional): group by 'bmes' when group_num=4, group by 'bme' when group_num = 3. Defaults to 3.
            bert_name (str): model name of bert series model, such as bert, roberta, xlnet, albert.
            blank_padding (bool, optional): whether pad sequence to max length. Defaults to True.
        """
        super(BERT_BMES_Lexicon_PinYin_Word_Attention_Cat_Encoder, self).__init__()

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

        self.group_num = group_num
        if self.group_num == 3:
            self.bmes2id = {'B': 0, 'M': 1, 'E': 2, '[PAD]': 3}
        else:
            self.bmes2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3, '[PAD]': 4}
        self.word2id = word2id
        self.pinyin2id = pinyin2id
        self.word_size = word_size
        self.lexicon_window_size = lexicon_window_size
        self.pinyin_size = pinyin_size
        self.max_length = max_length
        self.max_matched_lexcons = lexicon_window_size
        self.blank_padding = blank_padding
        # pinyin embedding matrix
        self.pinyin_embedding = nn.Embedding(len(self.pinyin2id), self.pinyin_size, padding_idx=self.pinyin2id['[PAD]'])
        if pinyin_embedding is not None:
            self.pinyin_embedding.weight.data.copy_(pinyin_embedding.weight.data)
            self.pinyin_embedding.weight.requires_grad = pinyin_embedding.weight.requires_grad
        # align word embedding and bert embedding
        self.bmes_lexicon_pinyin2bert = nn.Linear(len(self.bmes2id) + self.word_size + self.pinyin_size, self.bert.config.hidden_size)
        self.hidden_size = self.bert.config.hidden_size + len(self.bmes2id) + self.pinyin_size + self.word_size


    def forward(self, seqs_token_ids, seqs_lexicon_embed, seqs_pinyin_ids, seqs_lexicon_bmes_ids, att_lexicon_mask, att_token_mask):
        """
        Args:
            seqs: (B, L), index of tokens
            att_token_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        if 'roberta' in self.bert_name:
            # seq_out = self.bert(seqs, attention_mask=att_token_mask)[1][1] # hfl roberta
            bert_seqs_embed, _ = self.bert(seqs_token_ids, attention_mask=att_token_mask) # clue-roberta
        else:
            bert_seqs_embed, _ = self.bert(seqs_token_ids, attention_mask=att_token_mask)

        bmes_one_hot_embed = torch.zeros(*(seqs_lexicon_bmes_ids.size() + (len(self.bmes2id), ))).to(seqs_lexicon_bmes_ids.device)
        bmes_one_hot_embed.scatter_(-1, seqs_lexicon_bmes_ids.unsqueeze(-1), 1)
        bmes_one_hot_embed[seqs_lexicon_bmes_ids == self.bmes2id['[PAD]']] = 0.
        seqs_pinyin_embed = self.pinyin_embedding(seqs_pinyin_ids)
        cat_embed = torch.cat([bmes_one_hot_embed, seqs_lexicon_embed, seqs_pinyin_embed], dim=-1)
        cat_embed_att_output, _ = dot_product_attention_with_project(bert_seqs_embed, cat_embed, att_lexicon_mask, self.bmes_lexicon_pinyin2bert)
        inputs_embed = torch.cat([bert_seqs_embed, cat_embed_att_output], dim=-1) # (B, L, EMBED)

        return inputs_embed
    

    def lexicon_match(self, tokens):
        """
        Args:
            tokens (list): token list
        Returns:
            indexed_bmes (list[list]): index of tokens' segmentation label in matched words.
            indexed lexicons (list[list]): encode ids of tokens' matched words.
            indexed pinyins (list[list]): encode ids of tokens' pinyin in matched words.
        """
        indexed_bmes = [[self.bmes2id['B'] if self.group_num == 3 else self.bmes2id['S']] + [self.bmes2id['[PAD]']] * (self.max_matched_lexcons - 1)]
        indexed_lexicons = [[self.word2id['[CLS]']] + [self.word2id['[PAD]']] * (self.max_matched_lexcons - 1)]
        indexed_pinyins = [[self.pinyin2id['[UNK]']] + [self.pinyin2id['[PAD]']] * (self.max_matched_lexcons - 1)]
        for i in range(len(tokens)):
            words = []
            indexed_bmes.append([])
            indexed_lexicons.append([])
            indexed_pinyins.append([])
            for w in range(self.lexicon_window_size, 1 if self.group_num == 3 else 0, -1):
                for p in range(w):
                    if i - p < 0:
                        break
                    if i - p + w > len(tokens):
                        continue
                    word = ''.join(tokens[i-p:i-p+w])
                    if word in self.word2id and word not in '～'.join(words):
                        words.append(word)
                        # try:
                        #     pinyin = lazy_pinyin(word, style=Style.TONE3, nuetral_tone_with_five=True)[p]
                        #     if len(pinyin) > 7:
                        #         if pinyin.isnumeric():
                        #             pinyin = '[DIGIT]'
                        #         else:
                        #             pinyin = strip_accents(pinyin)
                        #             if pinyin.encode('utf-8').isalpha():
                        #                 pinyin = '[ENG]'
                        #             else:
                        #                 raise ValueError('pinyin length not exceed 7')
                        try:
                            pinyin = lazy_pinyin(word, style=Style.TONE3, neutral_tone_with_five=True)[p]
                            if len(pinyin) > 7:
                                if is_digit(pinyin):
                                    pinyin = '[DIGIT]'
                                else:
                                    pinyin = strip_accents(pinyin)
                                    if is_eng_word(pinyin):
                                        pinyin = '[ENG]'
                                    else:
                                        raise ValueError('pinyin length not exceed 7')
                            elif not is_pinyin(pinyin):
                                pinyin = '[UNK]'
                        except:
                            pinyin = '[UNK]'
                        if w == 1:
                            g == 'S'
                        elif p == 0:
                            g = 'B'
                        elif p == w - 1:
                            g = 'E'
                        else:
                            g = 'M'
                        indexed_bmes[-1].append(self.bmes2id[g])
                        indexed_lexicons[-1].append(self.word2id[word])
                        indexed_pinyins[-1].append(self.pinyin2id[pinyin] if pinyin in self.pinyin2id else self.pinyin2id['[UNK]'])
            indexed_bmes[-1].extend([self.bmes2id['[PAD]']] * (self.max_matched_lexcons - len(indexed_bmes[-1])))
            indexed_lexicons[-1].extend([self.word2id['[PAD]']] * (self.max_matched_lexcons - len(indexed_lexicons[-1])))
            indexed_pinyins[-1].extend([self.pinyin2id['[PAD]']] * (self.max_matched_lexcons - len(indexed_pinyins[-1])))
        indexed_bmes.append([self.bmes2id['B'] if self.group_num == 3 else self.bmes2id['S']] + [self.bmes2id['[PAD]']] * (self.max_matched_lexcons - 1))
        indexed_lexicons.append([self.word2id['[SEP]']] + [self.word2id['[PAD]']] * (self.max_matched_lexcons - 1))
        indexed_pinyins.append([self.pinyin2id['[UNK]']] + [self.pinyin2id['[PAD]']] * (self.max_matched_lexcons - 1))
        return indexed_bmes, indexed_lexicons, indexed_pinyins


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
                indexed_bmes, indexed_lexicons, indexed_pinyins = self.lexicon_match(tokens[1:-1])
                for _ in range(self.max_length - len(tokens)):
                    indexed_bmes.append([self.bmes2id['[PAD]']] * self.max_matched_lexcons)
                    indexed_lexicons.append([self.word2id['[PAD]']] * self.max_matched_lexcons)
                    indexed_pinyins.append([self.pinyin2id['[PAD]']] * self.max_matched_lexcons)
            else:
                indexed_tokens[self.max_length - 1] = indexed_tokens[-1]
                indexed_tokens = indexed_tokens[:self.max_length]
                indexed_bmes, indexed_lexicons, indexed_pinyins = self.lexicon_match(tokens[1:self.max_length - 1])
                is_truncated = True
        else:
            indexed_bmes, indexed_lexicons, indexed_pinyins = self.lexicon_match(tokens[1:-1])
            
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)
        indexed_lexicons = torch.tensor(indexed_lexicons).long().unsqueeze(0) # (1, L, W)
        indexed_pinyins = torch.tensor(indexed_pinyins).long().unsqueeze(0) # (1, L, W)
        indexed_bmes = torch.tensor(indexed_bmes).long().unsqueeze(0) # (1, L, W)
        # attention mask
        att_token_mask = (indexed_tokens != self.tokenizer.convert_tokens_to_ids('[PAD]')).type(torch.uint8) # (1, L)
        att_lexicon_mask = (indexed_lexicons != self.word2id['[PAD]']).type(torch.uint8) # (1, L, 3, W)

        # ensure the first three is indexed_tokens and indexed_lexicons and indexed_token2pinyins, the last is att_token_mask
        return indexed_tokens, indexed_lexicons, indexed_pinyins, indexed_bmes, att_lexicon_mask, att_token_mask  



class BERT_BMES_Lexicon_PinYin_Word_Attention_Add_Encoder(BERT_BMES_Lexicon_PinYin_Word_Attention_Cat_Encoder):
    def __init__(self, 
                pretrain_path,
                word2id,
                pinyin2id,
                pinyin_embedding=None,
                word_size=50,
                lexicon_window_size=4,
                pinyin_size=50,
                max_length=512, 
                group_num=3,
                bert_name='bert', 
                blank_padding=True):
        super().__init__(
            pretrain_path,
            word2id,
            pinyin2id,
            pinyin_embedding,
            word_size,
            lexicon_window_size,
            pinyin_size,
            max_length,
            group_num,
            bert_name, 
            blank_padding
        )
        self.hidden_size = self.bert.config.hidden_size

    def forward(self, seqs_token_ids, seqs_lexicon_embed, seqs_pinyin_ids, seqs_lexicon_bmes_ids, att_lexicon_mask, att_token_mask):
        """
        Args:
            seqs: (B, L), index of tokens
            att_token_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        if 'roberta' in self.bert_name:
            # seq_out = self.bert(seqs, attention_mask=att_token_mask)[1][1] # hfl roberta
            bert_seqs_embed, _ = self.bert(seqs_token_ids, attention_mask=att_token_mask) # clue-roberta
        else:
            bert_seqs_embed, _ = self.bert(seqs_token_ids, attention_mask=att_token_mask)

        bmes_one_hot_embed = torch.zeros(*(seqs_lexicon_bmes_ids.size() + (len(self.bmes2id), ))).to(seqs_lexicon_bmes_ids.device)
        bmes_one_hot_embed.scatter_(-1, seqs_lexicon_bmes_ids.unsqueeze(-1), 1)
        seqs_pinyin_embed = self.pinyin_embedding(seqs_pinyin_ids)
        cat_embed = self.bmes_lexicon_pinyin2bert(torch.cat([bmes_one_hot_embed, seqs_lexicon_embed, seqs_pinyin_embed], dim=-1))
        cat_embed_att_output, _ = dot_product_attention(bert_seqs_embed, cat_embed, att_lexicon_mask)
        inputs_embed = torch.add(bert_seqs_embed, cat_embed_att_output) # (B, L, EMBED)

        return inputs_embed



class BERT_BMES_Lexicon_PinYin_Char_Attention_Cat_Encoder(nn.Module):
    def __init__(self, 
                pretrain_path,
                word2id,
                pinyin_char2id,
                word_size=50,
                lexicon_window_size=4,
                pinyin_char_size=50,
                max_length=512, 
                max_pinyin_char_length=7,
                group_num=3,
                bert_name='bert', 
                blank_padding=True):
        """
        Args:
            pretrain_path (str): path of pretrain model.
            word2id (dict): dictionary of word -> idx mapping.
            pinyin_char2id (dict): dictionary of pinyin character -> idx mapping.
            word_size (int, optional): size of word embedding. Defaults to 50.
            lexicon_window_size (int, optional): upper bound(include) of lexicon match window size. Defaults to 4.
            max_length (int, optional): max length of sentence, used for postion embedding. Defaults to 512.
            max_pinyin_char_length (int, optional): max character length of a pinyin. Defaults to 7.
            group_num (int, optional): group by 'bmes' when group_num=4, group by 'bme' when group_num = 3. Defaults to 3.
            bert_name (str): model name of bert series model, such as bert, roberta, xlnet, albert.
            blank_padding (bool, optional): whether pad sequence to max length. Defaults to True.
        """
        super(BERT_BMES_Lexicon_PinYin_Char_Attention_Cat_Encoder, self).__init__()

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

        self.group_num = 3
        if self.group_num == 3:
            self.bmes2id = {'B': 0, 'M': 1, 'E': 2, '[UNK]': 3}
        else:
            self.bmes2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3, '[UNK]': 4}
        self.word2id = word2id
        self.pinyin_char2id = pinyin_char2id
        self.word_size = word_size
        self.lexicon_window_size = lexicon_window_size
        self.pinyin_char_size = pinyin_char_size
        self.max_length = max_length
        # self.max_matched_lexcons = (1 + lexicon_window_size) * lexicon_window_size - 1 - 2 * (lexicon_window_size - 1)
        self.max_matched_lexcons = lexicon_window_size
        self.max_pinyin_char_length = max_pinyin_char_length
        self.blank_padding = blank_padding
        # pinyin character embedding matrix
        self.pinyin_char_embedding = nn.Embedding(len(self.pinyin_char2id), self.pinyin_char_size, padding_idx=self.pinyin_char2id['[PAD]'])
        # align word embedding and bert embedding
        self.char_conv = nn.Conv1d(self.pinyin_char_size, self.pinyin_char_size * 2, kernel_size=3, padding=1)
        self.masked_conv1d = masked_singlekernel_conv1d
        self.bmes_lexicon_pinyin2bert = nn.Linear(len(self.bmes2id) + self.word_size + self.pinyin_char_size * 2, self.bert.config.hidden_size)
        self.hidden_size = self.bert.config.hidden_size + len(self.bmes2id) + self.word_size + self.pinyin_char_size * 2


    def forward(self, seqs_token_ids, seqs_lexicon_embed, seqs_pinyin_char_ids, seqs_lexicon_bmes_ids, att_pinyin_char_mask, att_lexicon_mask, att_token_mask):
        """
        Args:
            seqs: (B, L), index of tokens
            att_token_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        if 'roberta' in self.bert_name:
            # seq_out = self.bert(seqs, attention_mask=att_token_mask)[1][1] # hfl roberta
            bert_seqs_embed, _ = self.bert(seqs_token_ids, attention_mask=att_token_mask) # clue-roberta
        else:
            bert_seqs_embed, _ = self.bert(seqs_token_ids, attention_mask=att_token_mask)
        
        bmes_one_hot_embed = torch.zeros(*(seqs_lexicon_bmes_ids.size() + (len(self.bmes2id), ))).to(seqs_lexicon_bmes_ids.device)
        bmes_one_hot_embed.scatter_(-1, seqs_lexicon_bmes_ids.unsqueeze(-1), 1)
        seqs_pinyin_char_embed = self.pinyin_char_embedding(seqs_pinyin_char_ids)
        pinyin_conv = self.masked_conv1d(seqs_pinyin_char_embed, att_pinyin_char_mask, self.char_conv)
        cat_embed = torch.cat([bmes_one_hot_embed, seqs_lexicon_embed, pinyin_conv], dim=-1)
        cat_embed_att_output, _ = dot_product_attention_with_project(bert_seqs_embed, cat_embed, att_lexicon_mask, self.bmes_lexicon_pinyin2bert)
        inputs_embed = torch.cat([bert_seqs_embed, cat_embed_att_output], dim=-1)

        return inputs_embed
    

    def lexicon_match(self, tokens):
        indexed_bmes = [[self.bmes2id['B'] if self.group_num == 3 else self.bmes2id['S']] * self.max_matched_lexcons]
        indexed_lexicons = [[self.word2id['[CLS]']] + [self.word2id['[PAD]']] * (self.max_matched_lexcons - 1)]
        indexed_pinyins_chars = [[[self.pinyin_char2id['[PAD]']] * self.max_pinyin_char_length] * self.max_matched_lexcons]
        for i in range(len(tokens)):
            words = []
            indexed_bmes.append([])
            indexed_lexicons.append([])
            indexed_pinyins_chars.append([])
            for w in range(self.lexicon_window_size, 1 if self.group_num == 3 else 0, -1):
                for p in range(w):
                    if i - p < 0:
                        break
                    if i - p + w > len(tokens):
                        continue
                    word = ''.join(tokens[i-p:i-p+w])
                    if word in self.word2id and word not in '～'.join(words):
                        words.append(word)
                        try:
                            pinyin = lazy_pinyin(word, style=Style.TONE3, neutral_tone_with_five=True)[p]
                            if len(pinyin) > 7:
                                raise ValueError('pinyin length not exceed 7')
                            elif not is_pinyin(pinyin) and not is_eng_word(pinyin):
                                pinyin = '[UNK]'
                        except:
                            pinyin = '[UNK]'
                        if w == 1:
                            g = 'S'
                        elif p == 0:
                            g = 'B'
                        elif p == w - 1:
                            g = 'E'
                        else:
                            g = 'M'
                        indexed_bmes[-1].append(self.bmes2id[g])
                        indexed_lexicons[-1].append(self.word2id[word])
                        if pinyin != '[UNK]':
                            indexed_pinyins_chars[-1].append([self.pinyin_char2id[pc] if pc in self.pinyin_char2id else self.pinyin_char2id['[UNK]'] for pc in pinyin] + [self.pinyin_char2id['[PAD]']] * (self.max_pinyin_char_length - len(pinyin)))
                        else:
                            indexed_pinyins_chars[-1].append([self.pinyin_char2id['[PAD]']] * self.max_pinyin_char_length)
            if len(indexed_lexicons[-1]) == 0:
                indexed_bmes[-1].append(self.bmes2id['[UNK]'])
                indexed_lexicons[-1].append(self.word2id['[UNK]'])
                indexed_pinyins_chars[-1].append([self.pinyin_char2id['[PAD]']] * self.max_pinyin_char_length)
            indexed_bmes[-1].extend([self.bmes2id['[UNK]']] * (self.max_matched_lexcons - len(indexed_bmes[-1])))
            indexed_lexicons[-1].extend([self.word2id['[PAD]']] * (self.max_matched_lexcons - len(indexed_lexicons[-1])))
            indexed_pinyins_chars[-1].extend([[self.pinyin_char2id['[PAD]']] * self.max_pinyin_char_length] * (self.max_matched_lexcons - len(indexed_pinyins_chars[-1])))
        indexed_bmes.append([self.bmes2id['B'] if self.group_num == 3 else self.bmes2id['S']] * self.max_matched_lexcons)
        indexed_lexicons.append([self.word2id['[SEP]']] + [self.word2id['[PAD]']] * (self.max_matched_lexcons - 1))
        indexed_pinyins_chars.append([[self.pinyin_char2id['[PAD]']] * self.max_pinyin_char_length] * self.max_matched_lexcons)
        return indexed_bmes, indexed_lexicons, indexed_pinyins_chars


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
                indexed_bmes, indexed_lexicons, indexed_pinyins_chars = self.lexicon_match(tokens[1:-1])
                for _ in range(self.max_length - len(tokens)):
                    indexed_bmes.append([self.bmes2id['[UNK]']] * self.max_matched_lexcons)
                    indexed_lexicons.append([self.word2id['[PAD]']] * self.max_matched_lexcons)
                    indexed_pinyins_chars.append([[self.pinyin_char2id['[PAD]']] * self.max_pinyin_char_length] * self.max_matched_lexcons)
            else:
                indexed_tokens[self.max_length - 1] = indexed_tokens[-1]
                indexed_tokens = indexed_tokens[:self.max_length]
                indexed_bmes, indexed_lexicons, indexed_pinyins_chars = self.lexicon_match(tokens[1:self.max_length - 1])
        else:
            indexed_bmes, indexed_lexicons, indexed_pinyins_chars = self.lexicon_match(tokens[1:-1])
        
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)
        indexed_lexicons = torch.tensor(indexed_lexicons).long().unsqueeze(0) # (1, L, W)
        indexed_pinyins_chars = torch.tensor(indexed_pinyins_chars).long().unsqueeze(0) # (1, L, W, P)
        indexed_bmes = torch.tensor(indexed_bmes).long().unsqueeze(0) # (1, L, W)
        # attention mask
        att_token_mask = (indexed_tokens != self.tokenizer.convert_tokens_to_ids('[PAD]')).type(torch.uint8) # (1, L)
        att_lexicon_mask = (indexed_lexicons != self.word2id['[PAD]']).type(torch.uint8) # (1, L, W)
        att_pinyin_char_mask = (indexed_pinyins_chars != self.pinyin_char2id['[PAD]']).type(torch.uint8) # (1, L, W, P)

        # ensure the first two is indexed_tokens and indexed_pinyin_chars, the last is att_token_mask
        return indexed_tokens, indexed_lexicons, indexed_pinyins_chars, indexed_bmes, att_pinyin_char_mask, att_lexicon_mask, att_token_mask  



class BERT_BMES_Lexicon_PinYin_Char_Attention_Add_Encoder(BERT_BMES_Lexicon_PinYin_Char_Attention_Cat_Encoder):
    def __init__(self, 
                pretrain_path,
                word2id,
                pinyin_char2id,
                word_size=50,
                lexicon_window_size=4,
                pinyin_char_size=50,
                max_length=512, 
                max_pinyin_char_length=7,
                group_num=3,
                bert_name='bert', 
                blank_padding=True):
        super().__init__(
            pretrain_path,
            word2id,
            pinyin_char2id,
            word_size,
            lexicon_window_size,
            pinyin_char_size,
            max_length,
            max_pinyin_char_length,
            group_num,
            bert_name, 
            blank_padding
        )
        self.hidden_size = self.bert.config.hidden_size


    def forward(self, seqs_token_ids, seqs_lexicon_embed, seqs_pinyin_char_ids, seqs_lexicon_bmes_ids, att_pinyin_char_mask, att_lexicon_mask, att_token_mask):
        if 'roberta' in self.bert_name:
            # seq_out = self.bert(seqs, attention_mask=att_token_mask)[1][1] # hfl roberta
            bert_seqs_embed, _ = self.bert(seqs_token_ids, attention_mask=att_token_mask) # clue-roberta
        else:
            bert_seqs_embed, _ = self.bert(seqs_token_ids, attention_mask=att_token_mask)
        bmes_one_hot_embed = torch.zeros(*(seqs_lexicon_bmes_ids.size() + (len(self.bmes2id), ))).to(seqs_lexicon_bmes_ids.device)
        bmes_one_hot_embed.scatter_(-1, seqs_lexicon_bmes_ids.unsqueeze(-1), 1)
        seqs_pinyin_char_embed = self.pinyin_char_embedding(seqs_pinyin_char_ids)
        pinyin_conv = self.masked_conv1d(seqs_pinyin_char_embed, att_pinyin_char_mask, self.char_conv)
        cat_embed = self.bmes_lexicon_pinyin2bert(torch.cat([bmes_one_hot_embed, seqs_lexicon_embed, pinyin_conv], dim=-1))
        cat_embed_att_output, _ = dot_product_attention(bert_seqs_embed, cat_embed, att_lexicon_mask)
        inputs_embed = torch.add(bert_seqs_embed, cat_embed_att_output)
        return inputs_embed


class BERT_BMES_Lexicon_PinYin_Char_MultiConv_Attention_Cat_Encoder(BERT_BMES_Lexicon_PinYin_Char_Attention_Cat_Encoder):
    def __init__(self, 
                pretrain_path,
                word2id,
                pinyin_char2id,
                word_size=50,
                lexicon_window_size=4,
                pinyin_char_size=50,
                max_length=512, 
                max_pinyin_char_length=7,
                group_num=3,
                bert_name='bert', 
                blank_padding=True,
                convs_config=[(100, 2), (100, 3), (100, 4)]):
        super().__init__(
            pretrain_path,
            word2id,
            pinyin_char2id,
            word_size,
            lexicon_window_size,
            pinyin_char_size,
            max_length,
            max_pinyin_char_length,
            group_num,
            bert_name, 
            blank_padding
        )
        self.char_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=self.pinyin_char_size, out_channels=oc, kernel_size=ks),
                nn.MaxPool1d(kernel_size=self.max_pinyin_char_length - ks + 1),
                nn.ReLU()
            )
            for oc, ks in convs_config
        ])
        self.masked_conv1d = masked_multikernel_conv1d
        pinyin_conv_size = sum(cc[0] for cc in convs_config)
        self.bmes_lexicon_pinyin2bert = nn.Linear(len(self.bmes2id) + self.word_size + pinyin_conv_size, self.bert.config.hidden_size)
        self.hidden_size = self.bert.config.hidden_size + len(self.bmes2id) + self.word_size + pinyin_conv_size



class BERT_BMES_Lexicon_PinYin_Char_MultiConv_Attention_Add_Encoder(BERT_BMES_Lexicon_PinYin_Char_Attention_Add_Encoder):
    def __init__(self, 
                pretrain_path,
                word2id,
                pinyin_char2id,
                word_size=50,
                lexicon_window_size=4,
                pinyin_char_size=50,
                max_length=512, 
                max_pinyin_char_length=7,
                group_num=3,
                bert_name='bert', 
                blank_padding=True,
                convs_config=[(100, 2), (100, 3), (100, 4)]):
        super().__init__(
            pretrain_path,
            word2id,
            pinyin_char2id,
            word_size,
            lexicon_window_size,
            pinyin_char_size,
            max_length,
            max_pinyin_char_length,
            group_num,
            bert_name, 
            blank_padding
        )
        self.char_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=self.pinyin_char_size, out_channels=oc, kernel_size=ks),
                nn.MaxPool1d(kernel_size=self.max_pinyin_char_length - ks + 1),
                nn.ReLU()
            )
            for oc, ks in convs_config
        ])
        self.masked_conv1d = masked_multikernel_conv1d
        pinyin_conv_size = sum(cc[0] for cc in convs_config)
        self.bmes_lexicon_pinyin2bert = nn.Linear(len(self.bmes2id) + self.word_size + pinyin_conv_size, self.bert.config.hidden_size)
