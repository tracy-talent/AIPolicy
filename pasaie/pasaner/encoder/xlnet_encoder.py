"""
 Author: liujian
 Date: 2020-10-26 17:54:15
 Last Modified by: liujian
 Last Modified time: 2020-10-26 17:54:15
"""

import logging

import torch
import torch.nn as nn
from transformers import XLNetModel, XLNetTokenizer


class XLNetEncoder(nn.Module):
    def __init__(self, max_length, pretrain_path, blank_padding=True):
        """
        Args:
            max_length (int): max length of sequence
            pretrain_path (str): path of pretrain model
            blank_padding (bool, optional): whether pad sequence to max length. Defaults to True.
        """
        super(XLNetEncoder, self).__init__()

        # self.xlnet = AutoModelForCausalLM.from_pretrained(pretrain_path, output_hidden_states=True) # permute(1,0,2)
        self.xlnet = XLNetModel.from_pretrained(pretrain_path, mem_len=max_length)
        self.tokenizer = XLNetTokenizer.from_pretrained(pretrain_path)
        # add missed tokens in vocab.txt
        num_added_tokens = self.tokenizer.add_tokens(['“', '”', '—'])
        print(f"we have added {num_added_tokens} tokens ['“', '”', '—']")
        self.xlnet.resize_token_embeddings(len(self.tokenizer))

        self.hidden_size = self.xlnet.config.hidden_size
        self.max_length = max_length
        self.blank_padding = blank_padding

    def forward(self, seqs, token_type_ids, att_mask):
        """
        Args:
            seqs: (B, L), index of tokens
            token_type_ids: (B, L), token type ids
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        seq_out, _ = self.xlnet(input_ids=seqs, attention_mask=att_mask, token_type_ids=token_type_ids)
        # if self.bert_name == 'xlnet':
            # seq_out = seq_out.permute(1, 0, 2)
        return seq_out
    
    def tokenize(self, *items): # items = (tokens, spans, [attrs, optional])
        """
        Args:
            items: (tokens, tags) or (tokens, spans, atrrs) or (sentence)
        Returns:
            indexed_tokens (torch.tensor): tokenizer encode ids of tokens, (1, L)
            token_type_ids (torch.tensor): token type ids, (1, L)
            att_mask (torch.tensor): token mask ids, (1, L)
        """
        if isinstance(items[0], list) or isinstance(items[0], tuple):
            sentence = items[0]
            is_token = True
        else:
            sentence = items[0]
            is_token = False
        
        re_items = [[] for i in range(len(items))]
        if is_token:
            # re_items[0].append('▁')
            # re_items[1].append('O')
            # if len(re_items) > 2:
            #     re_items[2].append('null')
            # spos = 0
            # for i in range(len(items[0])):
            #     if items[1][i][0] == 'S':
            #         if i > spos:
            #             tokens = self.tokenizer.tokenize(''.join(items[0][spos:i]))[1:]
            #             re_items[0].extend(tokens)
            #             re_items[1].extend(['O'] * len(tokens))
            #             if len(re_items) > 2:
            #                 re_items[2].extend(['null'] * len(tokens))
            #         re_items[0].append(items[0][i])
            #         re_items[1].append(items[1][i])
            #         if len(re_items) > 2:
            #             re_items[2].append(items[2][i])
            #         spos = i + 1
            #     elif items[1][i][0] == 'B':
            #         if i > spos:
            #             tokens = self.tokenizer.tokenize(''.join(items[0][spos:i]))[1:]
            #             re_items[0].extend(tokens)
            #             re_items[1].extend(['O'] * len(tokens))
            #             if len(re_items) > 2:
            #                 re_items[2].extend(['null'] * len(tokens))
            #         spos = i
            #     elif items[1][i][0] == 'E':
            #         tokens = self.tokenizer.tokenize(''.join(items[0][spos:i+1]))[1:]
            #         re_items[0].extend(tokens)
            #         if len(tokens) == 1:
            #             re_items[1].append('S' if len(re_items) > 2 else f'S-{items[1][i][2:]}')
            #         else:
            #             re_items[1].append('B' if len(re_items) > 2 else f'B-{items[1][i][2:]}')
            #             re_items[1].extend(['M' if len(re_items) > 2 else f'M-{items[1][i][2:]}'] * (len(tokens) - 2))
            #             re_items[1].append('E' if len(re_items) > 2 else f'E-{items[1][i][2:]}')
            #         if len(re_items) > 2:
            #             re_items[2].extend([items[2][i]] * len(tokens))
            #         spos = i + 1
            # if spos < len(items[0]):
            #     tokens = self.tokenizer.tokenize(''.join(items[0][spos:len(items[0])]))[1:]
            #     re_items[0].extend(tokens)
            #     re_items[1].extend(['O'] * len(tokens))
            #     if len(re_items) > 2:
            #         re_items[2].extend(['null'] * len(tokens))
            # re_items[0].extend(['<sep>', '<cls>'])
            # re_items[1].extend(['O', 'O'])
            # if len(re_items) > 2:
            #     re_items.extend(['null', 'null'])
            # for i in range(len(items)):
            #     items[i].clear()
            #     items[i].extend(re_items[i])
            # tokens = items[0]
            items[0].insert(0, '▁')
            items[0].extend(['<sep>', '<cls>'])
            items[1].insert(0, 'O')
            items[1].extend(['O', 'O'])
            if len(items) > 2:
                items[2].insert(0, 'null')
                items[2].extend(['null', 'null'])
            tokens = items[0]
        else:
            tokens = self.tokenizer.tokenize(sentence).extend(['<sep>', '<cls>'])
        # print(tokens)
        
        if self.blank_padding:
            if len(tokens) < self.max_length:
                indexed_tokens = [self.tokenizer.convert_tokens_to_ids('<pad>')] * (self.max_length - len(tokens))
                token_type_ids = [3] * (self.max_length - len(tokens)) # 3 for <pad>
            else:
                index_tokens = []
                token_type_ids = []
        indexed_tokens.extend(self.tokenizer.convert_tokens_to_ids(tokens))
        token_type_ids.extend([0] * len(tokens))
        token_type_ids[-1] = 2 # 2 for <cls>
        avail_len = torch.tensor([len(indexed_tokens)])

        if self.blank_padding:
            if len(indexed_tokens) > self.max_length:
                indexed_tokens = indexed_tokens[:self.max_length - 2].extend(indexed_tokens[-2:])
                token_type_ids = token_type_ids[:self.max_length - 1].append(2)
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)
        token_type_ids = torch.tensor(token_type_ids).long().unsqueeze(0) # (1, L)

        # attention mask
        att_mask = torch.zeros(indexed_tokens.size(), dtype=torch.uint8) # (1, L)
        att_mask[0, -avail_len:] = 1

        return indexed_tokens, token_type_ids, att_mask  # ensure the first and last is indexed_tokens and att_mask