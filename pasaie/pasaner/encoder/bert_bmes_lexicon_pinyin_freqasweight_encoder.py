"""
 Author: liujian
 Date: 2021-01-19 23:38:22
 Last Modified by: liujian
 Last Modified time: 2021-01-19 23:38:22
"""

from ...tokenization.utils import convert_by_vocab, strip_accents
from ...utils.common import is_eng_word, is_digit, is_pinyin, is_punctuation
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
from pypinyin import lazy_pinyin, Style



class BERT_BMES_Lexicon_PinYin_Word_FreqAsWeight_Encoder(nn.Module):
    def __init__(self, 
                pretrain_path,
                word2freq,
                word2id,
                pinyin2id,
                pinyin_embedding=None,
                lexicon_window_size=4,
                word_size=50,
                pinyin_size=50,
                max_length=512, 
                group_num=3,
                bert_name='bert', 
                blank_padding=True):
        """
        Args:
            pretrain_path (str): path of pretrain model.
            word2freq (dict): dictionary of word->ferquency mapping.
            word2id (dict): dictionary of word->idx mapping.
            pinyin2id (dict): dictionary of pinyin->idx mapping.
            word_size (int, optional): size of word embedding. Defaults to 50.
            lexicon_window_size (int, optional): upper bound(include) of lexicon match window size. Defaults to 4.
            max_length (int, optional): max length of sentence, used for postion embedding. Defaults to 512.
            group_num (int, optional): group by 'bmes' when group_num=4, group by 'bme' when group_num = 3. Defaults to 3.
            bert_name (str): model name of bert series model, such as bert, roberta, xlnet, albert.
            blank_padding (bool, optional): whether pad sequence to max length. Defaults to True.
        """
        super(BERT_BMES_Lexicon_PinYin_Word_FreqAsWeight_Encoder, self).__init__()

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
        self.word2freq = word2freq
        self.word2id = word2id
        self.pinyin2id = pinyin2id
        self.word_size = word_size
        self.lexicon_window_size = lexicon_window_size
        self.pinyin_size = pinyin_size
        self.max_length = max_length
        self.max_matched_lexcons = lexicon_window_size - 2
        self.blank_padding = blank_padding
        # pinyin embedding matrix
        self.pinyin_embedding = nn.Embedding(len(self.pinyin2id), self.pinyin_size, padding_idx=self.pinyin2id['[PAD]'])
        if pinyin_embedding is not None:
            self.pinyin_embedding.weight.data.copy_(pinyin_embedding.weight.data)
            self.pinyin_embedding.weight.requires_grad = pinyin_embedding.weight.requires_grad
        self.hidden_size = self.bert.config.hidden_size + (self.word_size + self.pinyin_size) * self.group_num


    def forward(self, seqs_token_ids, seqs_lexicon_embed, seqs_pinyin_ids, lexicons_freq, att_token_mask):
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

        # weighted seqs_lexicon)_embed by lexicon_feq
        seqs_pinyin_embed = self.pinyin_embedding(seqs_pinyin_ids)
        lexicon_pinyin_embed = torch.cat([seqs_lexicon_embed, seqs_pinyin_embed], dim=-1)
        lexicons_freq_sum = lexicons_freq.sum(dim=(-2, -1), keepdim=True)
        lexicons_freq_sum[lexicons_freq_sum == 0] = 1 # avoid division by 0
        lexicon_pinyin_embed_weightedbyfreq = (lexicons_freq.unsqueeze(-1) * lexicon_pinyin_embed).sum(dim=-2) / lexicons_freq_sum
        flatten_emb_size = functools.reduce(lambda x, y: x * y, lexicon_pinyin_embed_weightedbyfreq.size()[-2:])
        lexicon_pinyin_embed_weightedbyfreq = lexicon_pinyin_embed_weightedbyfreq.view(*(lexicon_pinyin_embed_weightedbyfreq.size()[:-2] + (flatten_emb_size, )))
        inputs_embed = torch.cat([bert_seqs_embed, lexicon_pinyin_embed_weightedbyfreq], dim=-1)

        return inputs_embed
    
    
    def lexicon_match(self, tokens):
        lexicons_freq = [[[0] * self.max_matched_lexcons] * self.group_num]
        indexed_lexicons = [[[self.word2id['[CLS]']] + [self.word2id['[PAD]']] * (self.max_matched_lexcons - 1)] * self.group_num]
        indexed_pinyins = [[[self.pinyin2id['[UNK]']] + [self.pinyin2id['[PAD]']] * (self.max_matched_lexcons - 1)] * self.group_num]
        for i in range(len(tokens)):
            words = []
            lexicons_freq.append([[] for _ in range(self.group_num)])
            indexed_lexicons.append([[] for _ in range(self.group_num)])
            indexed_pinyins.append([[] for _ in range(self.group_num)])
            for w in range(self.lexicon_window_size, 1 if self.group_num == 3 else 0, -1):
                for p in range(w):
                    if i - p < 0:
                        break
                    if i - p + w > len(tokens):
                        continue
                    word = ''.join(tokens[i-p:i-p+w])
                    if word in self.word2id and word not in '～'.join(words):
                        words.append(word)
                        pinyin = lazy_pinyin(word, style=Style.TONE3, nuetral_tone_with_five=True)[p]
                        if is_digit(pinyin):
                            pinyin = '[DIGIT]'
                        elif is_eng_word(pinyin):
                            pinyin = '[ENG]'
                        elif is_pinyin(word) or is_punctuation(pinyin):
                            pinyin = pinyin
                        else:
                            pinyin = '[UNK]'

                        if w == 1:
                            g = 3
                        elif p == 0:
                            g = 0
                        elif p == w - 1:
                            g = 2
                        else:
                            g = 1
                        lexicons_freq[-1][g].append(self.word2freq[word] if word in self.word2freq else 0)
                        indexed_lexicons[-1][g].append(self.word2id[word])
                        indexed_pinyins[-1][g].append(self.pinyin2id[pinyin] if pinyin in self.pinyin2id else self.pinyin2id['[UNK]'])
            for g in range(self.group_num):
                if len(indexed_lexicons[-1][g]) == 0:
                    lexicons_freq[-1][g].append(0)
                    indexed_lexicons[-1][g].append(self.word2id['[UNK]'])
                    indexed_pinyins[-1][g].append(self.pinyin2id['[UNK]'])
                lexicons_freq[-1][g].extend([0] * (self.max_matched_lexcons - len(lexicons_freq[-1][g])))
                indexed_lexicons[-1][g].extend([self.word2id['[PAD]']] * (self.max_matched_lexcons - len(indexed_lexicons[-1][g])))
                indexed_pinyins[-1][g].extend([self.pinyin2id['[PAD]']] * (self.max_matched_lexcons - len(indexed_pinyins[-1][g])))
        lexicons_freq.append([[0] * self.max_matched_lexcons] * self.group_num)
        indexed_lexicons.append([[self.word2id['[SEP]']] + [self.word2id['[PAD]']] * (self.max_matched_lexcons - 1)] * self.group_num)
        indexed_pinyins.append([[self.pinyin2id['[UNK]']] + [self.pinyin2id['[PAD]']] * (self.max_matched_lexcons - 1)] * self.group_num)
        return lexicons_freq, indexed_lexicons, indexed_pinyins


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
                lexicons_freq, indexed_lexicons, indexed_pinyins = self.lexicon_match(tokens[1:-1])
                for _ in range(self.max_length - len(tokens)):
                    lexicons_freq.append([[0] * self.max_matched_lexcons] * self.group_num)
                    indexed_lexicons.append([[self.word2id['[PAD]']] * self.max_matched_lexcons] * self.group_num)
                    indexed_pinyins.append([[self.pinyin2id['[UNK]']] * self.max_matched_lexcons] * self.group_num)
            else:
                indexed_tokens[self.max_length - 1] = indexed_tokens[-1]
                indexed_tokens = indexed_tokens[:self.max_length]
                lexicons_freq, indexed_lexicons, indexed_pinyins = self.lexicon_match(tokens[1:self.max_length - 1])
        else:
            lexicons_freq, indexed_lexicons, indexed_pinyins = self.lexicon_match(tokens[1:-1])
            
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)
        lexicons_freq = torch.tensor(lexicons_freq).long().unsqueeze(0) # (1, L, 3, W)
        indexed_lexicons = torch.tensor(indexed_lexicons).long().unsqueeze(0) # (1, L, 3, W)
        indexed_pinyins = torch.tensor(indexed_pinyins).long().unsqueeze(0) # (1, L, 3, W)
        # attention mask
        att_token_mask = (indexed_tokens != self.tokenizer.convert_tokens_to_ids('[PAD]')).type(torch.uint8) # (1, L)

        # ensure the first three is indexed_tokens and indexed_lexicons and indexed_token2pinyins, the last is att_token_mask
        return indexed_tokens, indexed_lexicons, indexed_pinyins, lexicons_freq, att_token_mask  


class BERT_BMES_Lexicon_PinYin_Char_FreqAsWeight_Encoder(nn.Module):
    def __init__(self, 
                pretrain_path,
                word2freq,
                word2id,
                pinyin_char2id,
                word_size=50,
                lexicon_window_size=4,
                pinyin_char_size=50,
                max_pinyin_char_length=7,
                max_length=512, 
                group_num=3,
                bert_name='bert', 
                blank_padding=True):
        """
        Args:
            pretrain_path (str): path of pretrain model.
            word2freq (dict): dictionary of word->ferquency mapping.
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
        super(BERT_BMES_Lexicon_PinYin_Char_FreqAsWeight_Encoder, self).__init__()

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
        self.word2freq = word2freq
        self.word2id = word2id
        self.pinyin_char2id = pinyin_char2id
        self.word_size = word_size
        self.lexicon_window_size = lexicon_window_size
        self.max_matched_lexcons = lexicon_window_size - 2
        self.pinyin_char_size = pinyin_char_size
        self.max_pinyin_char_length = max_pinyin_char_length
        self.max_length = max_length
        self.blank_padding = blank_padding
        # pinyin character embedding matrix
        self.pinyin_char_embedding = nn.Embedding(len(self.pinyin_char2id), self.pinyin_char_size, padding_idx=self.pinyin_char2id['[PAD]'])
        self.char_conv = nn.Conv1d(self.pinyin_char_size, self.pinyin_char_size * 2, kernel_size=3, padding=1)
        self.masked_conv1d = masked_singlekernel_conv1d
        self.hidden_size = self.bert.config.hidden_size + (self.word_size + self.pinyin_char_size * 2) * self.group_num
        

    def forward(self, seqs_token_ids, seqs_lexicon_embed, seqs_pinyin_char_ids, att_pinyin_char_mask, lexicons_freq, att_token_mask):
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

        seqs_pinyin_char_embed = self.pinyin_char_embedding(seqs_pinyin_char_ids)
        pinyin_conv = self.masked_conv1d(seqs_pinyin_char_embed, att_pinyin_char_mask, self.char_conv)
        lexicon_pinyin_embed = torch.cat([seqs_lexicon_embed, pinyin_conv], dim=-1)
        lexicons_freq_sum = lexicons_freq.sum(dim=(-2, -1), keepdim=True)
        lexicons_freq_sum[lexicons_freq_sum == 0] = 1 # avoid division by 0
        lexicon_pinyin_embed_weightedbyfreq = (lexicons_freq.unsqueeze(-1) * lexicon_pinyin_embed).sum(dim=-2) / lexicons_freq_sum
        flatten_emb_size = functools.reduce(lambda x, y: x * y, lexicon_pinyin_embed_weightedbyfreq.size()[-2:])
        lexicon_pinyin_embed_weightedbyfreq = lexicon_pinyin_embed_weightedbyfreq.view(*(lexicon_pinyin_embed_weightedbyfreq.size()[:-2] + (flatten_emb_size, )))
        inputs_embed = torch.cat([bert_seqs_embed, lexicon_pinyin_embed_weightedbyfreq], dim=-1)

        return inputs_embed
    

    def lexicon_match(self, tokens):
        lexicons_freq = [[[0] * self.max_matched_lexcons] * self.group_num]
        indexed_lexicons = [[[self.word2id['[CLS]']] + [self.word2id['[PAD]']] * (self.max_matched_lexcons - 1)] * self.group_num]
        indexed_pinyins_chars = [[[[self.pinyin_char2id['[PAD]']] * self.max_pinyin_char_length] * self.max_matched_lexcons] * self.group_num]
        for i in range(len(tokens)):
            words = []
            lexicons_freq.append([[] for _ in range(self.group_num)])
            indexed_lexicons.append([[] for _ in range(self.group_num)])
            indexed_pinyins_chars.append([[] for _ in range(self.group_num)])
            for w in range(self.lexicon_window_size, 1 if self.group_num == 3 else 0, -1):
                for p in range(w):
                    if i - p < 0:
                        break
                    if i - p + w > len(tokens):
                        continue
                    word = ''.join(tokens[i-p:i-p+w])
                    if word in self.word2id and word not in '～'.join(words):
                    # if word in self.word2id:
                        words.append(word)
                        try:
                            pinyin = lazy_pinyin(word, style=Style.TONE3, neutral_tone_with_five=True)[p]
                            if len(pinyin) > 7:
                                raise ValueError('pinyin length not exceed 7')
                            elif not is_pinyin(pinyin) and not is_eng_word(pinyin) and not pinyin.isdigit():
                                pinyin = '[UNK]'
                        except:
                            pinyin = '[UNK]'
                        if w == 1:
                            g = 3
                        elif p == 0:
                            g = 0
                        elif p == w - 1:
                            g = 2
                        else:
                            g = 1
                        lexicons_freq[-1][g].append(self.word2freq[word] if word in self.word2freq else 0)
                        indexed_lexicons[-1][g].append(self.word2id[word])
                        if pinyin != '[UNK]':
                            indexed_pinyins_chars[-1][g].append([self.pinyin_char2id[pc] if pc in self.pinyin_char2id else self.pinyin_char2id['[UNK]'] for pc in pinyin] + [self.pinyin_char2id['[PAD]']] * (self.max_pinyin_char_length - len(pinyin)))
                        else:
                            indexed_pinyins_chars[-1][g].append([self.pinyin_char2id['[PAD]']] * self.max_pinyin_char_length)
            for g in range(self.group_num):
                if len(indexed_lexicons[-1][g]) == 0:
                    lexicons_freq[-1][g].append(0)
                    indexed_lexicons[-1][g].append(self.word2id['[UNK]'])
                    indexed_pinyins_chars[-1][g].append([self.pinyin_char2id['[PAD]']] * self.max_pinyin_char_length)
                lexicons_freq[-1][g].extend([0] * (self.max_matched_lexcons - len(lexicons_freq[-1][g])))
                indexed_lexicons[-1][g].extend([self.word2id['[PAD]']] * (self.max_matched_lexcons - len(indexed_lexicons[-1][g])))
                indexed_pinyins_chars[-1][g].extend([[self.pinyin_char2id['[PAD]']] * self.max_pinyin_char_length] * (self.max_matched_lexcons - len(indexed_pinyins_chars[-1][g])))
        lexicons_freq.append([[0] * self.max_matched_lexcons] * self.group_num)
        indexed_lexicons.append([[self.word2id['[SEP]']] + [self.word2id['[PAD]']] * (self.max_matched_lexcons - 1)] * self.group_num)
        indexed_pinyins_chars.append([[[self.pinyin_char2id['[PAD]']] * self.max_pinyin_char_length] * self.max_matched_lexcons] * self.group_num)
        return lexicons_freq, indexed_lexicons, indexed_pinyins_chars


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
                lexicons_freq, indexed_lexicons, indexed_pinyins_chars = self.lexicon_match(tokens[1:-1])
                for _ in range(self.max_length - len(tokens)):
                    lexicons_freq.append([[0] * self.max_matched_lexcons] * self.group_num)
                    indexed_lexicons.append([[self.word2id['[PAD]']] * self.max_matched_lexcons] * self.group_num)
                    indexed_pinyins_chars.append([[[self.pinyin_char2id['[PAD]']] * self.max_pinyin_char_length] * self.max_matched_lexcons] * self.group_num)
            else:
                indexed_tokens[self.max_length - 1] = indexed_tokens[-1]
                indexed_tokens = indexed_tokens[:self.max_length]
                lexicons_freq, indexed_lexicons, indexed_pinyins_chars = self.lexicon_match(tokens[1:self.max_length - 1])
                is_truncated = True
        else:
            lexicons_freq, indexed_lexicons, indexed_pinyins_chars = self.lexicon_match(tokens[1:-1])
            
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)
        indexed_lexicons = torch.tensor(indexed_lexicons).long().unsqueeze(0) # (1, L, g, W)
        indexed_pinyins_chars = torch.tensor(indexed_pinyins_chars).long().unsqueeze(0) # (1, L, 3, W, P)
        lexicons_freq = torch.tensor(lexicons_freq).long().unsqueeze(0) # (1, L, g, W)
        # attention mask
        att_token_mask = (indexed_tokens != self.tokenizer.convert_tokens_to_ids('[PAD]')).type(torch.uint8) # (1, L)
        att_pinyin_char_mask = (indexed_pinyins_chars != self.pinyin_char2id['[PAD]']).type(torch.uint8) # (1, L, g, W, P)

        # ensure the first two is indexed_tokens and indexed_token2word, the last is att_token_mask
        return indexed_tokens, indexed_lexicons, indexed_pinyins_chars, att_pinyin_char_mask, lexicons_freq, att_token_mask  



class BERT_BMES_Lexicon_PinYin_Char_MultiConv_FreqAsWeight_Encoder(BERT_BMES_Lexicon_PinYin_Char_FreqAsWeight_Encoder):
    def __init__(self, 
                pretrain_path,
                word2freq,
                word2id,
                pinyin_char2id,
                word_size=50,
                lexicon_window_size=4,
                pinyin_char_size=50,
                max_pinyin_char_length=7,
                max_length=512, 
                group_num=3,
                bert_name='bert', 
                blank_padding=True,
                convs_config=[(100, 2), (100, 3), (100, 4)]):
        super(BERT_BMES_Lexicon_PinYin_Char_MultiConv_FreqAsWeight_Encoder, self).__init__(
            pretrain_path=pretrain_path,
            word2freq=word2freq,
            word2id=word2id,
            pinyin_char2id=pinyin_char2id,
            word_size=word_size,
            lexicon_window_size=lexicon_window_size,
            pinyin_char_size=pinyin_char_size,
            max_pinyin_char_length=max_pinyin_char_length,
            max_length=max_length, 
            group_num=group_num,
            bert_name=bert_name, 
            blank_padding=blank_padding
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
        self.hidden_size = self.bert.config.hidden_size + (self.word_size + pinyin_conv_size) * self.group_num
