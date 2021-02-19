"""
 Author: liujian
 Date: 2021-02-06 14:37:32
 Last Modified by: liujian
 Last Modified time: 2021-02-06 14:37:32
"""

from ...tokenization.utils import strip_accents
from ...tokenization import WordTokenizer
from ...utils.common import is_eng_word, is_digit, is_pinyin
from ...module.nn.attention import dot_product_attention, dot_product_attention_with_project, MultiHeadedAttention
from ...module.nn.cnn import masked_singlekernel_conv1d, masked_multikernel_conv1d

import logging
import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer, AutoModel
from pypinyin import lazy_pinyin, Style


class BASE_Bigram_BMES_Lexicon_PinYin_Word_Attention_Cat_Encoder(nn.Module):
    def __init__(self, 
                unigram2id,
                bigram2id,
                word2id,
                pinyin2id,
                unigram_embedding=None,
                pinyin_embedding=None,
                unigram_size=50,
                bigram_size=50,
                word_size=50,
                lexicon_window_size=4,
                pinyin_size=50,
                max_length=512, 
                group_num=3,
                blank_padding=True,
                compress_seq=True):
        """
        Args:
            unigram2id (dict): dictionary of unigram->idx mapping.
            bigram2id (dict): dictionary of bigram->idx mapping.
            word2id (dict): dictionary of word->idx mapping.
            pinyin2id (dict): dictionary of pinyin->idx mapping.
            unigram_embedding (nn.Embedding): unigram embedding. Defaults to None.
            pinyin_embedding (nn.Embedding): pinyin embedding. Defaults to None.
            unigram_size (int, optional): size of unigram embedding. Defaults to 50.
            bigram_size (int, optional): size of bigram embedding. Defaults to 50.
            word_size (int, optional): size of word embedding. Defaults to 50.
            lexicon_window_size (int, optional): upper bound(include) of lexicon match window size. Defaults to 4.
            pinyin_size (int, optional): pinyin embedding size. Defaults to 50.
            max_length (int, optional): max length of sentence, used for postion embedding. Defaults to 512.
            group_num (int, optional): group by 'bmes' when group_num=4, group by 'bme' when group_num = 3. Defaults to 3.
            blank_padding (bool, optional): whether pad sequence to max length. Defaults to True.
            compress_seq (bool, optional): whether compress sequence before feed into LSTM. Defaults to True.
        """
        super(BASE_Bigram_BMES_Lexicon_PinYin_Word_Attention_Cat_Encoder, self).__init__()

        self.group_num = group_num
        if self.group_num == 3:
            self.bmes2id = {'B': 0, 'M': 1, 'E': 2, '[UNK]': 3}
        else:
            self.bmes2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3, '[UNK]': 4}
        self.unigram2id = unigram2id
        self.bigram2id = bigram2id
        self.word2id = word2id
        self.pinyin2id = pinyin2id
        self.unigram_size = unigram_size
        self.bigram_size = bigram_size
        self.word_size = word_size
        self.lexicon_window_size = lexicon_window_size
        self.pinyin_size = pinyin_size
        self.max_length = max_length
        self.max_matched_lexcons = lexicon_window_size
        self.blank_padding = blank_padding
        self.compress_seq = compress_seq
        # unigram embedding matrix
        self.unigram_embedding = nn.Embedding(len(self.unigram2id), self.unigram_size, padding_idx=self.unigram2id['[PAD]'])
        if unigram_embedding is not None:
            self.unigram_embedding.weight.data.copy_(unigram_embedding.weight.data)
            self.unigram_embedding.weight.requires_grad = unigram_embedding.weight.requires_grad
        # pinyin embedding matrix
        self.pinyin_embedding = nn.Embedding(len(self.pinyin2id), self.pinyin_size, padding_idx=self.pinyin2id['[PAD]'])
        if pinyin_embedding is not None:
            self.pinyin_embedding.weight.data.copy_(pinyin_embedding.weight.data)
            self.pinyin_embedding.weight.requires_grad = pinyin_embedding.weight.requires_grad
        # Transformer
        self.transformer = MultiHeadedAttention(
            d_input=self.unigram_size + self.bigram_size, 
            num_heads=8, 
            d_model=200, 
            d_ffn=200 * 3, 
            dropout_rate=0.1
        )
        self.bmes_lexicon_pinyin2gram = nn.Linear(len(self.bmes2id) + self.pinyin_size + self.word_size, 200)
        # Tokenizer
        self.tokenizer = WordTokenizer(vocab=self.unigram2id, unk_token="[UNK]")
        # hidden size of encoder output
        self.hidden_size = 200 + len(self.bmes2id) + self.pinyin_size + self.word_size
        #self.hidden_size = 200 + len(self.bmes2id) + self.pinyin_size + self.word_size

    def forward(self, seqs_unigram_ids, seqs_lexicon_embed, seqs_pinyin_ids, seqs_lexicon_bmes_ids, seqs_bigram_embed, att_lexicon_mask, att_unigram_mask):
        """
        Args:
            seqs: (B, L), index of tokens
            att_unigram_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        seqs_unigram_embed = self.unigram_embedding(seqs_unigram_ids)
        seqs_gram_embed = torch.cat([seqs_unigram_embed, seqs_bigram_embed], dim=-1)
        seqs_gram_embed = F.dropout(seqs_gram_embed, 0.5)
        seqs_gram_hidden = self.transformer(seqs_gram_embed, seqs_gram_embed, seqs_gram_embed, mask=att_unigram_mask)
        bmes_one_hot_embed = torch.zeros(*(seqs_lexicon_bmes_ids.size() + (len(self.bmes2id), ))).to(seqs_lexicon_bmes_ids.device)
        bmes_one_hot_embed.scatter_(-1, seqs_lexicon_bmes_ids.unsqueeze(-1), 1)
        seqs_pinyin_embed = self.pinyin_embedding(seqs_pinyin_ids)
        # cat_embed = torch.cat([bmes_one_hot_embed, seqs_lexicon_embed, seqs_pinyin_embed], dim=-1)
        cat_embed = F.dropout(torch.cat([bmes_one_hot_embed, seqs_lexicon_embed, seqs_pinyin_embed], dim=-1), 0.5)
        cat_embed_att_output, _ = dot_product_attention_with_project(seqs_gram_hidden, cat_embed, att_lexicon_mask, self.bmes_lexicon_pinyin2gram)
        inputs_embed = torch.cat([seqs_gram_hidden, cat_embed_att_output], dim=-1)

        return inputs_embed
    
    
    def lexicon_match(self, tokens):
        indexed_bmes = []
        indexed_lexicons = []
        indexed_pinyins = []
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
                        try:
                            pinyin = lazy_pinyin(word, style=Style.TONE3, nuetral_tone_with_five=True)[p]
                            if len(pinyin) > 7:
                                if pinyin.isnumeric():
                                    pinyin = '[DIGIT]'
                                else:
                                    pinyin = strip_accents(pinyin)
                                    if pinyin.encode('utf-8').isalpha():
                                        pinyin = '[ENG]'
                                    else:
                                        raise ValueError('pinyin length not exceed 7')
                        except:
                            pinyin = '[UNK]'
                        if w == 0:
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
            if len(indexed_lexicons[-1]) == 0:
                indexed_bmes[-1].append(self.bmes2id['[UNK]'])
                indexed_lexicons[-1].append(self.word2id['[UNK]'])
                indexed_pinyins[-1].append(self.pinyin2id['[UNK]'])
            indexed_bmes[-1].extend([self.bmes2id['[UNK]']] * (self.max_matched_lexcons - len(indexed_bmes[-1])))
            indexed_lexicons[-1].extend([self.word2id['[PAD]']] * (self.max_matched_lexcons - len(indexed_lexicons[-1])))
            indexed_pinyins[-1].extend([self.pinyin2id['[PAD]']] * (self.max_matched_lexcons - len(indexed_pinyins[-1])))
        return indexed_bmes, indexed_lexicons, indexed_pinyins


    def tokenize(self, *items): # items = (tokens, spans, [attrs, optional])
        """
        Args:
            items: (tokens, tags) or (tokens, spans, atrrs) or (sentence)
        Returns:
            indexed_unigrams (torch.tensor): tokenizer encode ids of tokens, (1, L)
            att_token_mask (torch.tensor): token mask ids, (1, L)
        """
        if isinstance(items[0], str):
            sentence = items[0]
            is_token = False
        else:
            sentence = items[0]
            is_token = True
        if is_token:
            unigrams = sentence
        else:
            unigrams = self.tokenizer.tokenize(sentence)
        indexed_unigrams = self.tokenizer.convert_tokens_to_ids(unigrams, blank_id=self.unigram2id['[PAD]'], unk_id=self.unigram2id['[UNK]'])
        indexed_bigrams = []
        for i in range(len(unigrams) - 1):
            bigram = ''.join(unigrams[i:i+2])
            if bigram in self.bigram2id:
                indexed_bigrams.append(self.bigram2id[bigram])
            else:
                indexed_bigrams.append(self.bigram2id['[UNK]'])
        indexed_bigrams.append(self.bigram2id['[UNK]'])
        if self.blank_padding:
            if len(indexed_unigrams) <= self.max_length:
                unigram_padding_idx = self.unigram2id['[PAD]']
                bigram_padding_idx = self.bigram2id['[PAD]']
                while len(indexed_unigrams) < self.max_length:
                    indexed_unigrams.append(unigram_padding_idx)
                    indexed_bigrams.append(bigram_padding_idx)
                indexed_bmes, indexed_lexicons, indexed_pinyins = self.lexicon_match(unigrams)
                for _ in range(self.max_length - len(unigrams)):
                    indexed_bmes.append([self.bmes2id['[UNK]']] * self.max_matched_lexcons)
                    indexed_lexicons.append([self.word2id['[PAD]']] * self.max_matched_lexcons)
                    indexed_pinyins.append([self.pinyin2id['[UNK]']] * self.max_matched_lexcons)
            else:
                indexed_unigrams = indexed_unigrams[:self.max_length]
                indexed_bigrams = indexed_bigrams[:self.max_length]
                indexed_bmes, indexed_lexicons, indexed_pinyins = self.lexicon_match(unigrams[:self.max_length])
        else:
            indexed_bmes, indexed_lexicons, indexed_pinyins = self.lexicon_match(unigrams)
            
        indexed_unigrams = torch.tensor(indexed_unigrams).long().unsqueeze(0) # (1, L)
        indexed_bigrams = torch.tensor(indexed_bigrams).long().unsqueeze(0) # (1, L)
        indexed_lexicons = torch.tensor(indexed_lexicons).long().unsqueeze(0) # (1, L, W)
        indexed_pinyins = torch.tensor(indexed_pinyins).long().unsqueeze(0) # (1, L, W)
        indexed_bmes = torch.tensor(indexed_bmes).long().unsqueeze(0) # (1, L, W)
        # attention mask
        att_unigram_mask = (indexed_unigrams != self.unigram2id['[PAD]']).type(torch.uint8) # (1, L)
        att_lexicon_mask = (indexed_lexicons != self.word2id['[PAD]']).type(torch.uint8) # (1, L, 3, W)

        # ensure the first three is indexed_unigrams and indexed_lexicons and indexed_token2pinyins, the last is att_token_mask
        return indexed_unigrams, indexed_lexicons, indexed_pinyins, indexed_bmes, indexed_bigrams, att_lexicon_mask, att_unigram_mask  



class BASE_Bigram_BMES_Lexicon_PinYin_Word_Attention_Add_Encoder(BASE_Bigram_BMES_Lexicon_PinYin_Word_Attention_Cat_Encoder):
    def __init__(self, 
                unigram2id,
                bigram2id,
                word2id,
                pinyin2id,
                unigram_embedding=None,
                pinyin_embedding=None,
                unigram_size=50,
                bigram_size=50,
                word_size=50,
                lexicon_window_size=4,
                pinyin_size=50,
                max_length=512, 
                group_num=3,
                blank_padding=True,
                compress_seq=True):
        super().__init__(
            unigram2id,
            bigram2id,
            word2id,
            pinyin2id,
            unigram_embedding,
            pinyin_embedding,
            unigram_size,
            bigram_size,
            word_size,
            lexicon_window_size,
            pinyin_size,
            max_length,
            group_num,
            blank_padding,
            compress_seq
        )
        #self.hidden_size = self.unigram_size + self.bigram_size
        self.hidden_size = 200

    def forward(self, seqs_unigram_ids, seqs_lexicon_embed, seqs_pinyin_ids, seqs_lexicon_bmes_ids, seqs_bigram_embed, att_lexicon_mask, att_unigram_mask):
        """
        Args:
            seqs: (B, L), index of tokens
            att_unigram_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        seqs_unigram_embed = self.unigram_embedding(seqs_unigram_ids)
        seqs_gram_embed = torch.cat([seqs_unigram_embed, seqs_bigram_embed], dim=-1)
        seqs_gram_embed = F.dropout(seqs_gram_embed, 0.5)
        seqs_gram_hidden = self.transformer(seqs_gram_embed, seqs_gram_embed, seqs_gram_embed, mask=att_unigram_mask)
        bmes_one_hot_embed = torch.zeros(*(seqs_lexicon_bmes_ids.size() + (len(self.bmes2id), ))).to(seqs_lexicon_bmes_ids.device)
        bmes_one_hot_embed.scatter_(-1, seqs_lexicon_bmes_ids.unsqueeze(-1), 1)
        seqs_pinyin_embed = self.pinyin_embedding(seqs_pinyin_ids)
        # cat_embed = self.bmes_lexicon_pinyin2gram(torch.cat([bmes_one_hot_embed, seqs_lexicon_embed, seqs_pinyin_embed], dim=-1))
        cat_embed = self.bmes_lexicon_pinyin2gram(F.dropout(torch.cat([bmes_one_hot_embed, seqs_lexicon_embed, seqs_pinyin_embed], dim=-1), 0.5))
        cat_embed_att_output, _ = dot_product_attention(seqs_gram_hidden, cat_embed, att_lexicon_mask)
        inputs_embed = torch.add(seqs_gram_hidden, cat_embed_att_output)

        return inputs_embed



class BASE_Bigram_BMES_Lexicon_PinYin_Char_Attention_Cat_Encoder(nn.Module):
    def __init__(self, 
                unigram2id,
                bigram2id,
                word2id,
                pinyin_char2id,
                unigram_embedding=None,
                unigram_size=50,
                bigram_size=50,
                word_size=50,
                lexicon_window_size=4,
                pinyin_char_size=50,
                max_length=512, 
                max_pinyin_char_length=7,
                group_num=3,
                blank_padding=True,
                compress_seq=True):
        """
        Args:
            unigram2id (dict): dictionary of unigram->idx mapping.
            bigram2id (dict): dictionary of bigram->idx mapping.
            word2id (dict): dictionary of word->idx mapping.
            pinyin_char2id (dict): dictionary of pinyin character -> idx mapping.
            unigram_embedding (nn.Embedding): token embedding. Defaults to None.
            unigram_size (int, optional): size of unigram embedding. Defaults to 50.
            bigram_size (int, optional): size of bigram embedding. Defaults to 50.
            word_size (int, optional): size of word embedding. Defaults to 50.
            lexicon_window_size (int, optional): upper bound(include) of lexicon match window size. Defaults to 4.
            max_length (int, optional): max length of sentence, used for postion embedding. Defaults to 512.
            max_pinyin_char_length (int, optional): max character length of a pinyin. Defaults to 7.
            group_num (int, optional): group by 'bmes' when group_num=4, group by 'bme' when group_num = 3. Defaults to 3.
            blank_padding (bool, optional): whether pad sequence to max length. Defaults to True.
            compress_seq (bool, optional): whether compress sequence before feed into LSTM. Defaults to True.
        """
        super(BASE_Bigram_BMES_Lexicon_PinYin_Char_Attention_Cat_Encoder, self).__init__()

        self.group_num = 3
        if self.group_num == 3:
            self.bmes2id = {'B': 0, 'M': 1, 'E': 2, '[UNK]': 3}
        else:
            self.bmes2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3, '[UNK]': 4}
        self.unigram2id = unigram2id
        self.bigram2id = bigram2id
        self.word2id = word2id
        self.pinyin_char2id = pinyin_char2id
        self.unigram_size = unigram_size
        self.bigram_size = bigram_size
        self.word_size = word_size
        self.lexicon_window_size = lexicon_window_size
        self.pinyin_char_size = pinyin_char_size
        self.max_length = max_length
        self.max_matched_lexcons = lexicon_window_size
        self.max_pinyin_char_length = max_pinyin_char_length
        self.blank_padding = blank_padding
        self.compress_seq = compress_seq
        # token embedding matric
        self.unigram_embedding = nn.Embedding(len(self.unigram2id), self.unigram_size, padding_idx=self.unigram2id['[PAD]'])
        if unigram_embedding is not None:
            self.unigram_embedding.weight.data.copy_(unigram_embedding.weight.data)
            self.unigram_embedding.weight.requires_grad = unigram_embedding.weight.requires_grad
        # pinyin character embedding matrix
        self.pinyin_char_embedding = nn.Embedding(len(self.pinyin_char2id), self.pinyin_char_size, padding_idx=self.pinyin_char2id['[PAD]'])
        self.char_conv = nn.Conv1d(self.pinyin_char_size, self.pinyin_char_size * 2, kernel_size=3, padding=1)
        self.masked_conv1d = masked_singlekernel_conv1d
         # LSTM
        self.bilstm = nn.LSTM(input_size=self.unigram_size + self.bigram_size, 
                            hidden_size=self.unigram_size + self.bigram_size, 
                            num_layers=1, 
                            bidirectional=True, 
                            batch_first=True)
        self.bmes_lexicon_pinyin2gram = nn.Linear(len(self.bmes2id) + self.word_size + self.pinyin_char_size * 2, self.unigram_size + self.bigram_size)
        # Tokenizer
        self.tokenizer = WordTokenizer(vocab=self.unigram2id, unk_token="[UNK]")
        # hidden size of encoder output
        self.hidden_size = self.unigram_size + self.bigram_size + len(self.bmes2id) + self.word_size + self.pinyin_char_size * 2


    def forward(self, seqs_unigram_ids, seqs_lexicon_embed, seqs_pinyin_char_ids, seqs_lexicon_bmes_ids, seqs_bigram_embed, att_pinyin_char_mask, att_lexicon_mask, att_unigram_mask):
        """
        Args:
            seqs: (B, L), index of tokens
            att_token_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        seqs_unigram_embed = self.unigram_embedding(seqs_unigram_ids)
        seqs_gram_embed = torch.cat([seqs_unigram_embed, seqs_bigram_embed], dim=-1)
        if self.compress_seq:
            seqs_length = att_unigram_mask.sum(dim=-1).detach().cpu()
            seqs_gram_embed_packed = pack_padded_sequence(seqs_gram_embed, seqs_length, batch_first=True)
            seqs_gram_hidden_packed, _ = self.bilstm(seqs_gram_embed_packed)
            seqs_gram_hidden, _ = pad_packed_sequence(seqs_gram_hidden_packed, batch_first=True) # B, S, D
        else:
            seqs_gram_hidden, _ = self.bilstm(seqs_gram_embed)
        seqs_gram_hidden = torch.add(*seqs_gram_hidden.chunk(2, dim=-1))
        bmes_one_hot_embed = torch.zeros(*(seqs_lexicon_bmes_ids.size() + (len(self.bmes2id), ))).to(seqs_lexicon_bmes_ids.device)
        bmes_one_hot_embed.scatter_(-1, seqs_lexicon_bmes_ids.unsqueeze(-1), 1)
        seqs_pinyin_char_embed = self.pinyin_char_embedding(seqs_pinyin_char_ids)
        pinyin_conv = self.masked_conv1d(seqs_pinyin_char_embed, att_pinyin_char_mask, self.char_conv)
        cat_embed = torch.cat([bmes_one_hot_embed, seqs_lexicon_embed, pinyin_conv], dim=-1)
        cat_embed_att_output, _ = dot_product_attention_with_project(seqs_gram_hidden, cat_embed, att_lexicon_mask, self.bmes_lexicon_pinyin2token)
        inputs_embed = torch.cat([seqs_gram_hidden, cat_embed_att_output], dim=-1)

        return inputs_embed
    

    def lexicon_match(self, tokens):
        indexed_bmes = []
        indexed_lexicons = []
        indexed_pinyins_chars = []
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
        return indexed_bmes, indexed_lexicons, indexed_pinyins_chars


    def tokenize(self, *items): # items = (tokens, spans, [attrs, optional])
        """
        Args:
            items: (tokens, tags) or (tokens, spans, atrrs) or (sentence)
        Returns:
            indexed_unigrams (torch.tensor): tokenizer encode ids of tokens, (1, L)
            att_token_mask (torch.tensor): token mask ids, (1, L)
        """
        if isinstance(items[0], str):
            sentence = items[0]
            is_token = False
        else:
            sentence = items[0]
            is_token = True
        if is_token:
            unigrams = sentence
        else:
            unigrams = self.tokenizer.tokenize(sentence)

        indexed_unigrams = self.tokenizer.convert_tokens_to_ids(unigrams, blank_id=self.unigram2id['[PAD]'], unk_id=self.unigram2id['[UNK]'])
        indexed_bigrams = []
        for i in range(len(unigrams) - 1):
            bigram = ''.join(unigrams[i:i+2])
            if bigram in self.bigram2id:
                indexed_bigrams.append(self.bigram2id[bigram])
            else:
                indexed_bigrams.append(self.bigram2id['[UNK]'])
        indexed_bigrams.append(self.bigram2id['[UNK]'])
        if self.blank_padding:
            if len(indexed_unigrams) <= self.max_length:
                unigram_padding_idx = self.unigram2id['[PAD]']
                bigram_padding_idx = self.bigram2id['[PAD]']
                while len(indexed_unigrams) < self.max_length:
                    indexed_unigrams.append(unigram_padding_idx)
                    indexed_bigrams.append(bigram_padding_idx)
                indexed_bmes, indexed_lexicons, indexed_pinyins_chars = self.lexicon_match(unigrams)
                for _ in range(self.max_length - len(unigrams)):
                    indexed_bmes.append([self.bmes2id['[UNK]']] * self.max_matched_lexcons)
                    indexed_lexicons.append([self.word2id['[PAD]']] * self.max_matched_lexcons)
                    indexed_pinyins_chars.append([[self.pinyin_char2id['[PAD]']] * self.max_pinyin_char_length] * self.max_matched_lexcons)
            else:
                indexed_unigrams = indexed_unigrams[:self.max_length]
                indexed_bigrams = indexed_bigrams[:self.max_length]
                indexed_bmes, indexed_lexicons, indexed_pinyins_chars = self.lexicon_match(unigrams[:self.max_length])
        else:
            indexed_bmes, indexed_lexicons, indexed_pinyins_chars = self.lexicon_match(unigrams)
        
        indexed_unigrams = torch.tensor(indexed_unigrams).long().unsqueeze(0) # (1, L)
        indexed_bigrams = torch.tensor(indexed_bigrams).long().unsqueeze(0) # (1, L)
        indexed_lexicons = torch.tensor(indexed_lexicons).long().unsqueeze(0) # (1, L, W)
        indexed_pinyins_chars = torch.tensor(indexed_pinyins_chars).long().unsqueeze(0) # (1, L, W, P)
        indexed_bmes = torch.tensor(indexed_bmes).long().unsqueeze(0) # (1, L, W)
        # attention mask
        att_unigram_mask = (indexed_unigrams != self.unigram2id['[PAD]']).type(torch.uint8) # (1, L)
        att_lexicon_mask = (indexed_lexicons != self.word2id['[PAD]']).type(torch.uint8) # (1, L, W)
        att_pinyin_char_mask = (indexed_pinyins_chars != self.pinyin_char2id['[PAD]']).type(torch.uint8) # (1, L, W, P)

        # ensure the first two is indexed_unigrams and indexed_pinyin_chars, the last is att_token_mask
        return indexed_unigrams, indexed_lexicons, indexed_pinyins_chars, indexed_bmes, indexed_bigrams, att_pinyin_char_mask, att_lexicon_mask, att_unigram_mask



class BASE_Bigram_BMES_Lexicon_PinYin_Char_Attention_Add_Encoder(BASE_Bigram_BMES_Lexicon_PinYin_Char_Attention_Cat_Encoder):
    def __init__(self, 
                unigram2id,
                bigram2id,
                word2id,
                pinyin_char2id,
                unigram_embedding=None,
                unigram_size=50,
                bigram_size=50,
                word_size=50,
                lexicon_window_size=4,
                pinyin_char_size=50,
                max_length=512, 
                max_pinyin_char_length=7,
                group_num=3,
                blank_padding=True,
                compress_seq=True):
        super().__init__(
            unigram2id,
            bigram2id,
            word2id,
            pinyin_char2id,
            unigram_embedding,
            unigram_size,
            bigram_size,
            word_size,
            lexicon_window_size,
            pinyin_char_size,
            max_length,
            max_pinyin_char_length,
            group_num,
            blank_padding,
            compress_seq
        )
        self.hidden_size = self.unigram_size + self.bigram_size

    def forward(self, seqs_unigram_ids, seqs_lexicon_embed, seqs_pinyin_char_ids, seqs_lexicon_bmes_ids, seqs_bigram_embed, att_pinyin_char_mask, att_lexicon_mask, att_unigram_mask):
        """
        Args:
            seqs: (B, L), index of tokens
            att_token_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        seqs_unigram_embed = self.unigram_embedding(seqs_unigram_ids)
        seqs_gram_embed = torch.cat([seqs_unigram_embed, seqs_bigram_embed], dim=-1)
        if self.compress_seq:
            seqs_length = att_unigram_mask.sum(dim=-1).detach().cpu()
            seqs_gram_embed_packed = pack_padded_sequence(seqs_gram_embed, seqs_length, batch_first=True)
            seqs_gram_hidden_packed, _ = self.bilstm(seqs_gram_embed_packed)
            seqs_gram_hidden, _ = pad_packed_sequence(seqs_gram_hidden_packed, batch_first=True) # B, S, D
        else:
            seqs_gram_hidden, _ = self.bilstm(seqs_gram_embed)
        seqs_gram_hidden = torch.add(*seqs_gram_hidden.chunk(2, dim=-1))
        bmes_one_hot_embed = torch.zeros(*(seqs_lexicon_bmes_ids.size() + (len(self.bmes2id), ))).to(seqs_lexicon_bmes_ids.device)
        bmes_one_hot_embed.scatter_(-1, seqs_lexicon_bmes_ids.unsqueeze(-1), 1)
        seqs_pinyin_char_embed = self.pinyin_char_embedding(seqs_pinyin_char_ids)
        pinyin_conv = self.masked_conv1d(seqs_pinyin_char_embed, att_pinyin_char_mask, self.char_conv)
        cat_embed = self.bmes_lexicon_pinyin2gram(torch.cat([bmes_one_hot_embed, seqs_lexicon_embed, pinyin_conv], dim=-1))
        cat_embed_att_output, _ = dot_product_attention(seqs_gram_hidden, cat_embed, att_lexicon_mask)
        inputs_embed = torch.add(seqs_gram_hidden, cat_embed_att_output)

        return inputs_embed



class BASE_Bigram_BMES_Lexicon_PinYin_Char_MultiConv_Attention_Cat_Encoder(BASE_Bigram_BMES_Lexicon_PinYin_Char_Attention_Cat_Encoder):
    def __init__(self, 
                unigram2id,
                bigram2id,
                word2id,
                pinyin_char2id,
                unigram_embedding=None,
                unigram_size=50,
                bigram_size=50,
                word_size=50,
                lexicon_window_size=4,
                pinyin_char_size=50,
                max_length=512, 
                max_pinyin_char_length=7,
                group_num=3,
                convs_config=[(100, 2), (100, 3), (100, 4)],
                blank_padding=True,
                compress_seq=True):
        super().__init__(
            unigram2id,
            bigram2id,
            word2id,
            pinyin_char2id,
            unigram_embedding,
            unigram_size,
            bigram_size,
            word_size,
            lexicon_window_size,
            pinyin_char_size,
            max_length,
            max_pinyin_char_length,
            group_num,
            blank_padding,
            compress_seq)
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
        self.bmes_lexicon_pinyin2gram = nn.Linear(len(self.bmes2id) + self.word_size + pinyin_conv_size, self.unigram_size + self.bigram_size)
        self.hidden_size = self.unigram_size + self.bigram_size + len(self.bmes2id) + self.word_size + pinyin_conv_size



class BASE_Bigram_BMES_Lexicon_PinYin_Char_MultiConv_Attention_Add_Encoder(BASE_Bigram_BMES_Lexicon_PinYin_Char_Attention_Add_Encoder):
    def __init__(self, 
                unigram2id,
                bigram2id,
                word2id,
                pinyin_char2id,
                unigram_embedding=None,
                unigram_size=50,
                bigram_size=50,
                word_size=50,
                lexicon_window_size=4,
                pinyin_char_size=50,
                max_length=512, 
                max_pinyin_char_length=7,
                group_num=3,
                convs_config=[(100, 2), (100, 3), (100, 4)],
                blank_padding=True,
                compress_seq=True):
        super().__init__(
            unigram2id,
            bigram2id,
            word2id,
            pinyin_char2id,
            unigram_embedding,
            unigram_size,
            bigram_size,
            word_size,
            lexicon_window_size,
            pinyin_char_size,
            max_length,
            max_pinyin_char_length,
            group_num,
            blank_padding,
            compress_seq)
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
        self.bmes_lexicon_pinyin2gram = nn.Linear(len(self.bmes2id) + self.word_size + pinyin_conv_size, self.unigram_size + self.bigram_size)
        self.hidden_size = self.unigram_size + self.bigram_size
