"""
 Author: liujian
 Date: 2021-01-29 17:39:35
 Last Modified by: liujian
 Last Modified time: 2021-01-29 17:39:35
"""

import os
import json
from timedec import timeit
from collections import deque

from gensim.models import KeyedVectors

# 字典树，由语料匹配词典统计词频信息

class AC_Automaton(object):
    def __init__(self):
        self.trans = [dict() for _ in range(int(1e6))]
        self.cnt = [0 for _ in range(int(1e6))]
        self.fail = [0 for _ in range(int(1e6))] 
        self.deq = deque()
        self.tot = 0

    def insert(self, freq, tokens):
        """insert operation

        Args:
            freq (int): frequency of tokens occur.
            tokens (list): token list.
        """
        u = 0
        for token in tokens:
            if token not in self.trans[u]:
                self.tot += 1
                self.trans[u][token] = self.tot
            u = self.trans[u][token]
        self.cnt[u] += freq

    def insert(self, freq, tokens):
        node = self
        for token in tokens:
            if token in node.next:
                node = node.next[token]
                node.cnt += freq
            else:
                node.next[token] = TrieNode()
                node = node.next[token]
                node.cnt += freq



class WordFrequencyFromCorpus(object):
    def __init__(self, lexicon_path):
        """
        Args:
            lexicon_file (str): lexicon file path.
        """
        wv = KeyedVectors.load(lexicon_path)
        self.wordset = set(wv.vocab)
        self.trie_root = TrieNode()

    @timeit
    def word_frequency_statistics(self, data_path, lexicon_window_size=8, output_path=None):
        """
        Args:
            data_path (str): data path.
            lexicon_window_size (int, optional): lexicon window size decide max length of word. Defaults to 8.
            output_path (str, optional): output path. Defaults to None.

        Returns:
            [type]: [description]
        """
        word_freq = {}
        for ds in ['train', 'dev']:
            dataset_file = f'{data_path}/{ds}.char.bmoes'
            if not os.path.exists(dataset_file):
                continue
            with open(dataset_file, 'r', encoding='utf-8') as f:
                tokens = []
                for line in f:
                    line = line.strip().split()
                    if len(line) > 0:
                        tokens.append(line[0])
                    elif len(tokens) > 0:
                        for w in range(2, lexicon_window_size + 1):
                            for i in range(len(tokens) - w + 1):
                                word = ''.join(tokens[i:i+w])
                                if word in self.wordset:
                                    if word in word_freq:
                                        word_freq[word][0] += 1
                                    else:
                                        word_freq[word] = [1, tokens[i:i+w]]
                        words = []
        word_freq_naive = {w:f for w, (f, t) in word_freq.items()}
        with open(f'{data_path}/word_freq_total.json', 'w', encoding='utf-8') as of:
            json.dump(word_freq_naive, of, ensure_ascii=False)
        word_freq_tokens = sorted(word_freq.items(), key=lambda item: len(item[0]), reverse=True)
        word_freq = {}

        @timeit
        def stat():
            for word, (freq, tokens) in word_freq_tokens:
                is_subword, cnt = self.trie_root.query(tokens)
                if is_subword:
                    freq -= cnt
                self.trie_root.insert(freq, tokens)
                word_freq[word] = freq
        stat()
        self.trie_root = TrieNode()
        if output_path is None:
            output_path = f'{data_path}/word_freq.json'
        with open(output_path, 'w', encoding='utf-8') as of:
            json.dump(word_freq, of, ensure_ascii=False)                
        return word_freq


if __name__ == '__main__':
    lexicon_path = '/home/liujian/NLP/corpus/embedding/chinese/lexicon/ctbword_gigachar_mix.710k.50d.bin'
    data_path = '../../input/benchmark/entity/weibo'
    corpus_name = ['weibo', 'resume', 'ontontoes4', 'msra']
    wstats = WordFrequencyFromCorpus(lexicon_path=lexicon_path)
    wstats.word_frequency_statistics(data_path, lexicon_window_size=4)