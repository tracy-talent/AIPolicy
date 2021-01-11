"""
 Author: liujian
 Date: 2020-10-26 17:54:15
 Last Modified by: liujian
 Last Modified time: 2020-10-26 17:54:15
"""

from ...tokenization import JiebaTokenizer

import logging
import math

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import BertModel, AlbertModel, BertTokenizer
from transformers.modeling_bert import BertEmbeddings


class BERTWLFEncoder(nn.Module):
    def __init__(self, 
                pretrain_path,
                word2id,
                word_size=50,
                word2vec=None,
                custom_dict=None,
                max_length=512, 
                bert_name='bert', 
                blank_padding=True):
        """
        Args:
            word2id (dict): dictionary of word->idx mapping
            word_size (int, optional): size of word embedding. Defaults to 50.
            word2vec (numpy.ndarray, optional): pretrained word2vec numpy. Defaults to None.
            custom_dict (str, optional): customized dictionary for word tokenizer. Defaults to None.
            max_length (int, optional): max length of sentence, used for postion embedding. Defaults to 512.
            pretrain_path (str): path of pretrain model
            bert_name (str): model name of bert series model, such as bert, roberta, xlnet, albert
            blank_padding (bool, optional): whether pad sequence to max length. Defaults to True.
        """
        super(BERTWLFEncoder, self).__init__()

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

        # load word vectors
        self.word2id = word2id
        self.num_word = len(word2id)
        if word2vec is None:
            self.word_size = word_size
        else:
            self.word_size = word2vec.shape[-1]
        if word2vec is not None:
            try:
                word2vec = torch.from_numpy(word2vec)
            except TypeError as e:
                logging.info(e)
        # word vocab
        if not '[CLS]' in self.word2id:
            self.word2id['[CLS]'] = len(self.word2id)
            self.num_word += 1
            if word2vec is not None:
                cls_vec = torch.randn(1, self.word_size) / math.sqrt(self.word_size)
                word2vec = torch.cat([word2vec, cls_vec], dim=0)
        if not '[SEP]' in self.word2id:
            self.word2id['[SEP]'] = len(self.word2id)
            self.num_word += 1
            if word2vec is not None:
                sep_vec = torch.randn(1, self.word_size) / math.sqrt(self.word_size)
                word2vec = torch.cat([word2vec, sep_vec], dim=0)
        if not '[UNK]' in self.word2id:
            self.word2id['[UNK]'] = len(self.word2id)
            self.num_word += 1
            if word2vec is not None:
                unk_vec = torch.randn(1, self.word_size) / math.sqrt(self.word_size)
                word2vec = torch.cat([word2vec, unk_vec], dim=0)
        if not '[PAD]' in self.word2id:
            self.word2id['[PAD]'] = len(self.word2id)
            self.num_word += 1
            if word2vec is not None:
                blk_vec = torch.zeros(1, self.word_size)
                word2vec = torch.cat([word2vec, blk_vec], dim=0)
        # word embedding
        self.word_embedding = nn.Embedding(self.num_word, self.word_size)
        if word2vec is not None:
            logging.info("Initializing word embedding with word2vec.")
            self.word_embedding.weight.data.copy_(word2vec)
        # word tokenizer
        self.word_tokenizer = JiebaTokenizer(vocab=self.word2id, unk_token="[UNK]", custom_dict=custom_dict)

        # self.hidden_size = self.bert.config.hidden_size + self.word_size
        self.hidden_size = self.bert.config.hidden_size + self.word_size
        self.max_length = max_length
        self.blank_padding = blank_padding

        # align word embedding and bert embedding
        self.word2bert_linear = nn.Linear(self.word_size, self.word_size)


    def forward(self, seqs_char, seqs_word, att_mask):
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
        inputs_embed = torch.cat([
            bert_seq_embed,
            self.word2bert_linear(self.word_embedding(seqs_word.detach().cpu()).to(self.word2vert_linear.weight.device))
        ], dim=-1) # (B, L, EMBED)

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
            words = ['[CLS]'] + self.word_tokenizer.tokenize(''.join(sentence)) + ['[SEP]']
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
            words = ['[CLS]'] + self.word_tokenizer.tokenize(sentence) + ['[SEP]']
        
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        indexed_words = self.word_tokenizer.convert_tokens_to_ids(words, blank_id=self.word2id['[PAD]'], unk_id=self.word2id['[UNK]'])
        avail_len = torch.tensor([len(indexed_tokens)])
        token2word = [0] * avail_len
        wpos, wlen, tlen = 0, 0, 0
        for i in range(avail_len):
            if tlen >= wlen + len(words[wpos]):
                wlen += len(words[wpos])
                wpos += 1
            tlen += len(tokens[i])
            token2word[i] = wpos

        if self.blank_padding:
            is_truncated = False
            if len(indexed_tokens) <= self.max_length:
                while len(indexed_tokens) < self.max_length:
                    indexed_tokens.append(0)
            else:
                indexed_tokens[self.max_length - 1] = indexed_tokens[-1]
                indexed_tokens = indexed_tokens[:self.max_length]
                is_truncated = True
            indexed_token2word = [self.word2id['[PAD]']] * self.max_length
            for i in range(min(self.max_length, avail_len)):
                indexed_token2word[i] = indexed_words[token2word[i]]
            if is_truncated:
                indexed_token2word[self.max_length - 1] = self.word2id['[SEP]']
        else:
            indexed_token2word = [self.word2id['[PAD]']] * avail_len
            for i in range(avail_len):
                indexed_token2word[i] = indexed_words[token2word[i]]
            
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)
        indexed_token2word = torch.tensor(indexed_token2word).long().unsqueeze(0) # (1, L)
        # attention mask
        att_mask = torch.zeros(indexed_tokens.size(), dtype=torch.uint8) # (1, L)
        att_mask[0, :avail_len] = 1

        return indexed_tokens, indexed_token2word, att_mask  # ensure the first and last is indexed_tokens and att_mask



class MRC_BERTWLFEncoder(BERTWLFEncoder):
    def tokenize(self, *items): # items = (tokens, spans, query, [attrs, optional])
        """
        Args:
            items: (tokens, tags, query) or (tokens, spans, atrrs, query) or (sentence)
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
        
        query_tokens = self.tokenizer.tokenize(items[-1])
        if is_token:
            words = ['[CLS]'] + self.word_tokenizer.tokenize(items[-1]) + ['[SEP]'] + self.word_tokenizer.tokenize(''.join(sentence)) + ['[SEP]']
            tokens = ['[CLS]'] + query_tokens + ['[SEP]'] + items[0] + ['[SEP]']
            items[0].clear()
            items[0].extend(tokens)
            if len(items) > 2:
                spans = ['O'] + ['O'] * len(query_tokens) + ['O'] + items[1] + ['O']
                items[1].clear()
                items[1].extend(spans)
            if len(items) > 3:
                attrs = ['null'] + ['null'] * len(query_tokens) + ['null'] + items[2] + ['null']
                items[2].clear()
                items[2].extend(attrs)
        else:
            words = ['[CLS]'] + self.word_tokenizer.tokenize(items[-1]) + ['[SEP]'] + self.word_tokenizer.tokenize(sentence) + ['[SEP]']
            tokens = ['[CLS]'] + query_tokens + ['[SEP]'] + self.tokenizer.tokenize(sentence) + ['[SEP]']
        
        
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        indexed_words = self.word_tokenizer.convert_tokens_to_ids(words, blank_id=self.word2id('[PAD]'), unk_id=self.word2id('[UNK]'))
        avail_len = torch.tensor([len(indexed_tokens)])
        token2word = [0] * avail_len
        wpos, wlen, tlen = 0, 0, 0
        for i in range(avail_len):
            if tlen >= wlen + len(words[wpos]):
                wlen += len(words[wpos])
                wpos += 1
            tlen += len(tokens[i])
            token2word[i] = wpos

        if self.blank_padding:
            is_truncated = False
            if len(indexed_tokens) <= self.max_length:
                while len(indexed_tokens) < self.max_length:
                    indexed_tokens.append(0)
            else:
                indexed_tokens[self.max_length - 1] = indexed_tokens[-1]
                indexed_tokens = indexed_tokens[:self.max_length]
                is_truncated = True
            indexed_token2word = [self.word2id['[PAD]']] * self.max_length
            for i in range(min(self.max_length, avail_len)):
                indexed_token2word[i] = indexed_words[token2word[i]]
            if is_truncated:
                indexed_token2word[self.max_length - 1] = self.word2id['[SEP]']
        else:
            indexed_token2word = [self.word2id['[PAD]']] * avail_len
            for i in range(avail_len):
                indexed_token2word[i] = indexed_words[token2word[i]]
            
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)
        indexed_token2word = torch.tensor(indexed_token2word).long().unsqueeze(0) # (1, L)

        # attention mask
        att_mask = torch.zeros(indexed_tokens.size(), dtype=torch.uint8) # (1, L)
        att_mask[0, :avail_len] = 1
        loss_mask = torch.zeros(indexed_tokens.size(), dtype=torch.uint8) # (1, L)
        loss_mask[0, len(query_tokens)+2:avail_len-1] = 1

        return indexed_tokens, loss_mask, att_mask  # ensure the first and last is indexed_tokens and att_mask