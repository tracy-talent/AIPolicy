"""
 Author: liujian
 Date: 2021-01-29 17:39:35
 Last Modified by: liujian
 Last Modified time: 2021-01-29 17:39:35
"""

import os
import json
from timedec import timeit
from collections import deque, defaultdict, OrderedDict

from gensim.models import KeyedVectors


class WordFrequencyFromCorpus(object):
    def __init__(self, lexicon_path):
        """
        Args:
            lexicon_file (str): lexicon file path.
        """
        wv = KeyedVectors.load(lexicon_path)
        self.wordset = set(wv.vocab)

    @timeit
    def word_frequency_statistics(self, data_path, lexicon_name='ctb', lexicon_window_size=8, output_path=None):
        """
        Args:
            data_path (str): data path.
            lexicon_window_size (int, optional): lexicon window size decide max length of word. Defaults to 8.
            output_path (str, optional): output path. Defaults to None.

        Returns:
            [type]: [description]
        """
        if output_path is None:
            output_path = f'{data_path}/word_freq/{lexicon_name}'
        os.makedirs(output_path, exist_ok=True)
        word_freq = {}
        for ds in ['train', 'dev']:
            if 'msra' in data_path and ds == 'dev':
                continue
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
                        for w in range(2, lexicon_window_size + 1):
                            for i in range(len(tokens) - w + 1):
                                word = ''.join(tokens[i:i+w])
                                if word in self.wordset:
                                    if word in word_freq:
                                        word_freq[word][0] += 1
                                    else:
                                        word_freq[word] = [1, tokens[i:i+w]]
                        tokens = []
                        sent_num += 1
                        if (sent_num  + 1) % 1000 == 0:
                            print(f'processed {sent_num + 1} lines', flush=True)

        word_freq_tokens = sorted(word_freq.items(), key=lambda item: len(item[0]), reverse=True)
        word_freq_naive = {w:f for w, (f, t) in word_freq_tokens}
        with open(f'{output_path}/word_freq_naive_w2-{lexicon_window_size}.json', 'w', encoding='utf-8') as of:
            json.dump(word_freq_naive, of, ensure_ascii=False)
        word_freq_filter = OrderedDict()

        @timeit
        def stat(window, word_freq_filter, word_freq_tokens):
            ngrams_freq = defaultdict(lambda: 0)
            for w1, (freq1, tokens1) in word_freq_tokens:
                if len(tokens1) < window:
                    window -= 1
                    ngrams_freq = defaultdict(lambda: 0)
                    for w2, (freq2, tokens2) in word_freq_filter.items():
                        for i in range(len(tokens2) - window + 1):
                            word = ''.join(tokens2[i:i+window])
                            if word in self.wordset:
                                ngrams_freq[word] += freq2
                if w1 in ngrams_freq:
                    word_freq_filter[w1] = (freq1 - ngrams_freq[w1], tokens1)
                else:
                    word_freq_filter[w1] = (freq1, tokens1)
        stat(lexicon_window_size, word_freq_filter, word_freq_tokens)
        word_freq_filter = OrderedDict({k:f for k, (f, _) in word_freq_filter.items()})
        with open(f'{output_path}/word_freq_filter_w2-{lexicon_window_size}.json', 'w', encoding='utf-8') as of:
            json.dump(word_freq_filter, of, ensure_ascii=False)
        return word_freq_filter


if __name__ == '__main__':
    # lexicon_path = '/home/liujian/NLP/corpus/embedding/chinese/lexicon/ctbword_gigachar_mix.710k.50d.bin'
    lexicon_path = '/home/liujian/NLP/corpus/embedding/chinese/lexicon/sgns_merge_word.1293k.300d.bin'
    data_path = '../../input/benchmark/entity'
    # corpus_name = ['weibo', 'resume', 'ontonotes4', 'msra', 'policy']
    corpus_name = ['policy']
    wstats = WordFrequencyFromCorpus(lexicon_path=lexicon_path)
    for corpus in corpus_name:
        for w in range(4, 9):
            print('*' * 20 + f'process {corpus}, window_size: {w} ' + '*' * 20 + '\n')
            wstats.word_frequency_statistics(os.path.join(data_path, corpus), lexicon_window_size=w, lexicon_name='sgns')
