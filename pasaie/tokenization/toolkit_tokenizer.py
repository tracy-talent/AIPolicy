# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WordpieceTokenizer classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unicodedata
import jieba

from .utils import load_vocab, convert_by_vocab

class JiebaTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab = None, unk_token="[UNK]", custom_dict='../../input/token/custom_dict.txt'):
        self.vocab = load_vocab(vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.ids_to_tokens = self.inv_vocab
        self.unk_token = unk_token
        if len(custom_dict) > 0 and os.path.exists(custom_dict):
            jieba.load_userdict(custom_dict)

    def tokenize(self, text):
        """    Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform tokenization
            using the given vocabulary.

            For example:
                input = "unaffable"
                output = ["un", "##aff", "##able"]

            Args:
                text: A single token or whitespace separated tokens. This should have already been passed through `BasicTokenizer`.
            Returns:
                output_tokens: A list of wordpiece tokens.
                current_positions: A list of the current positions for the original words in text .
        """
        token_list = jieba.lcut(text, use_paddle=True)            
        return token_list

    def convert_tokens_to_ids(self, tokens, max_seq_length = None, blank_id = 0, unk_id = 1, uncased = True):
        return convert_by_vocab(self.vocab, tokens, max_seq_length, blank_id, unk_id, uncased=uncased)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)
