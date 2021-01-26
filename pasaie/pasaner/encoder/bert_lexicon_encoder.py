"""
 Author: liujian
 Date: 2021-01-24 21:33:13
 Last Modified by: liujian
 Last Modified time: 2021-01-24 21:33:13
"""

from ...module.nn.attention import dot_product_attention

import logging
import math

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import BertModel, AlbertModel, BertTokenizer
from transformers.modeling_bert import BertEmbeddings


class BERTLexiconEncoder(nn.Module):
    def __init__(self, 
                pretrain_path,
                word2id,
                word_size=50,
                lexicon_window_size=4,
                max_length=512, 
                bert_name='bert', 
                blank_padding=True):
        """
        Args:
            pretrain_path (str): path of pretrain model.
            word2id (dict): dictionary of word->idx mapping.
            word_size (int, optional): size of word embedding. Defaults to 50.
            lexicon_window_size (int, optional): upper bound(include) of lexicon match window size. Defaults to 4.
            max_length (int, optional): max length of sentence, used for postion embedding. Defaults to 512.
            bert_name (str): model name of bert series model, such as bert, roberta, xlnet, albert.
            blank_padding (bool, optional): whether pad sequence to max length. Defaults to True.
        """
        super(BERTLexiconEncoder, self).__init__()

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
        self.word_size = word_size
        self.lexicon_window_size = lexicon_window_size
        self.hidden_size = self.bert.config.hidden_size
        self.max_length = max_length
        self.max_matched_lexcions = (1 + lexicon_window_size) * lexicon_window_size - 1
        self.blank_padding = blank_padding
        # align word embedding and bert embedding
        self.lexicon2bert_linear = nn.Linear(self.word_size, self.bert.config.hidden_size)


    def forward(self, seqs_token_ids, seqs_lexcion_embed, att_lexicon_mask, att_token_mask):
        """
        Args:
            seqs: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        if 'roberta' in self.bert_name:
            # seq_out = self.bert(seqs, attention_mask=att_mask)[1][1] # hfl roberta
            bert_seq_embed, _ = self.bert(seqs_token_ids, attention_mask=att_token_mask) # clue-roberta
        else:
            bert_seq_embed, _ = self.bert(seqs_token_ids, attention_mask=att_token_mask)
        # seq_embedding = self.embeddings(seqs)
        # inputs_embed = torch.cat([
        #     bert_seq_embed,
        #     self.word2bert_linear(seqs_word_embed)
        # ], dim=-1) # (B, L, EMBED)
        lexicon2bert_embed = self.lexicon2bert_linear(seqs_lexcion_embed)
        lexicon_att_output, _ = dot_product_attention(bert_seq_embed, lexicon2bert_embed, att_lexicon_mask)
        inputs_embed = torch.add(bert_seq_embed, lexicon_att_output) # (B, L, EMBED)
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
        if self.blank_padding:
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
        else:
            indexed_lexicons = self.lexicon_match(tokens[1:-1])            
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)
        indexed_lexicons = torch.tensor(indexed_lexicons).long().unsqueeze(0) # (1, L, W)
        # attention mask
        att_token_mask = (indexed_tokens != self.tokenizer.convert_tokens_to_ids('[PAD]')).type(torch.uint8) # (1, L)
        att_lexicon_mask = (indexed_lexicons != self.word2id['[PAD]']).type(torch.uint8) # (1, L, W)
        
        # ensure the first two is indexed_tokens and indexed_lexicons, the last is att_mask
        return indexed_tokens, indexed_lexicons, att_lexicon_mask, att_token_mask  
