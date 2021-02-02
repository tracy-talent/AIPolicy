"""
 Author: liujian
 Date: 2021-01-29 17:39:35
 Last Modified by: liujian
 Last Modified time: 2021-01-29 17:39:35
"""

import os
import json
from timedec import timeit
from collections import defaultdict, OrderedDict

from gensim.models import KeyedVectors


class TrieNode(object):
    def __init__(self):
        self.next = {}
        self.isword = False


class Trie(object):
    def __init__(self):
        self.root = TrieNode()

    def insert(self, tokens):
        node = self.root
        for token in tokens:
            if token not in node.next:
                node.next[token] = TrieNode()
            node = node.next[token]
        node.isword = True
    
    def query(self, tokens):
        node = self.root
        for token in tokens:
            if token not in node.next:
                return False
            node = node.next[token]
        return node.isword


class WordFrequencyFromCorpus(object):
    def __init__(self, lexicon_path):
        """
        Args:
            lexicon_file (str): lexicon file path.
        """
        wv = KeyedVectors.load(lexicon_path)
        self.wordset = set(wv.vocab)
        self.trie = Trie()
        for word in self.wordset:
            self.trie.insert(word)


    def enumerate_matched(self, tokens, pos, max_ngram, min_ngram):
        matched_words = []
        for i in range(min_ngram, min(len(tokens), pos + max_ngram) + 1):
            if self.trie.query(tokens[pos:pos+i]):
                matched_words.append((pos, tokens[pos:pos+i]))
        return matched_words


    @timeit
    def word_frequency_statistics(self, data_path, lexicon_name='ctb', max_ngram='$', min_ngram=2, output_path=None):
        """
        Args:
            data_path (str): data path.
            max_ngram (int or str, optional): lexicon window size decide max length of word, '$' means no length limit. Defaults to 8.
            output_path (str, optional): output path. Defaults to None.
            unigram (bool, optional): the minimal ngram for frequencey statistics. Defaults to False.

        Returns:
            [type]: [description]
        """
        if output_path is None:
            output_path = f'{data_path}/word_freq/{lexicon_name}'
        os.makedirs(output_path, exist_ok=True)
        word_freq = defaultdict(lambda: 0)
        for ds in ['train', 'dev']:
            dataset_file = f'{data_path}/{ds}.char.bmoes'
            if not os.path.exists(dataset_file):
                continue
            with open(dataset_file, 'r', encoding='utf-8') as f:
                tokens = []
                sent_num = 0
                for line in f:
                    line = line.strip().split()
                    if len(line) > 0:
                        tokens.append(line[0])
                    elif len(tokens) > 0:
                        max_word_length = max_ngram
                        if max_ngram == '$':
                            max_word_length = int(1e6)
                        matched_pos_tokens_pairs = []
                        for i in range(len(tokens)):
                            matched_pos_tokens_pairs.extend(self.enumerate_matched(tokens, i, max_word_length, min_ngram))
                        matched_pos_tokens_pairs.sort(key=lambda x: -len(x[1]))
                        while matched_pos_tokens_pairs:
                            pos_tokens = matched_pos_tokens_pairs[0]
                            word_freq[''.join(pos_tokens[1])] += 1
                            for i in range(len(pos_tokens[1])):
                                for j in range(i + 1, len(pos_tokens[1]) + 1):
                                    if (pos_tokens[0] + i, pos_tokens[1][i:j]) in matched_pos_tokens_pairs:
                                        matched_pos_tokens_pairs.remove((pos_tokens[0] + i, pos_tokens[1][i:j]))
                        tokens = []
                        sent_num += 1
                        if (sent_num  + 1) % 1000 == 0:
                            print(f'processed {sent_num + 1} lines', flush=True)
        word_freq = OrderedDict(sorted(word_freq.items(), key=lambda x: -len(x[0])))
        with open(f'{output_path}/word_freq_w{min_ngram}-{max_ngram}.json', 'w', encoding='utf-8') as of:
            json.dump(word_freq, of, ensure_ascii=False)
        return word_freq


if __name__ == '__main__':
    data_path = '../../input/benchmark/entity'
    for lexicon in ['ctbword_gigachar_mix.710k.50d.bin', 'sgns_merge_word.1293k.300d.bin']:
        lexicon_path = f'/home/liujian/NLP/corpus/embedding/chinese/lexicon/{lexicon}'
        if 'ctb' in lexicon:
            lexicon_name = 'ctb'
        else:
            lexicon_name = 'sgns'
        wstats = WordFrequencyFromCorpus(lexicon_path=lexicon_path)
        max_ngrams = ['$'] + list(range(4, 9))
        for corpus_name in ['weibo', 'resume', 'ontonotes4', 'msra', 'policy']:
            for w in max_ngrams:
                for ngram in [1, 2]:
                    print('*' * 20 + f'process {corpus_name}, window_size: {w}, min_ngram: {ngram}' + '*' * 20 + '\n')
                    wstats.word_frequency_statistics(os.path.join(data_path, corpus_name), max_ngram=w, lexicon_name=lexicon_name, min_ngram=ngram)
    # lexicon_path = '/home/liujian/NLP/corpus/embedding/chinese/lexicon/ctbword_gigachar_mix.710k.50d.bin'
    # # lexicon_path = '/home/liujian/NLP/corpus/embedding/chinese/lexicon/sgns_merge_word.1293k.300d.bin'
    # # corpus_name = ['weibo', 'resume', 'ontonotes4', 'msra', 'policy']
    # corpus_name = ['weibo']
    # wstats = WordFrequencyFromCorpus(lexicon_path=lexicon_path)
    # for corpus in corpus_name:
    #     for w in range(4, 9):
    #         print('*' * 20 + f'process {corpus}, window_size: {w} ' + '*' * 20 + '\n')
    #         wstats.word_frequency_statistics(os.path.join(data_path, corpus), max_ngram=w, lexicon_name='ctb', min_ngram=1)
