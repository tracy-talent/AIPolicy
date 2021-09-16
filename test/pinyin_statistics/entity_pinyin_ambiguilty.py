# -*-coding:utf-8-*-
"""
@Time    : 2021/7/16 1:54
@Author  : Mecthew
@File    : entity_pinyin_ambiguilty.py
"""
from pypinyin import lazy_pinyin, Style, pinyin
from collections import defaultdict
import json
from ast import literal_eval


def lexicon_match(word2id, word2pinyin, tokens, entities, lexicon_window_size):
    matched_lexicon_num, polyphonic_lexicon_num = 0, 0
    polyphonic_entity_num = 0

    for entity in entities:
        pos_b, pos_e = entity[0]

        for i in [pos_b, pos_e-1]:
            words = []
            pinyins = []
            start = 0 if dataset == 'weibo' else 1
            for w in range(lexicon_window_size, start, -1):
                for p in range(w):
                    if i - p < 0:
                        break
                    if i - p + w > len(tokens):
                        continue
                    word = ''.join(tokens[i - p: i - p + w])
                    if word in word2id:
                        matched_lexicon_num += 1
                        words.append(word)
                        # print(lazy_pinyin(word, style=Style.TONE3, neutral_tone_with_five=True)[p])
                        try:
                            if len(word) == 1:
                                # if word in word2pinyin:
                                #     pinyins.extend(word2pinyin[word])
                                # else:
                                #     pinyins.append(word)
                                pinyins.extend(pinyin(word, style=Style.TONE3, heteronym=True)[0])
                            else:
                                pinyins.append(lazy_pinyin(list(word), style=Style.TONE3, neutral_tone_with_five=True)[p])
                        except:
                            print(lazy_pinyin(list(word), style=Style.TONE3, neutral_tone_with_five=True), p)
            if len(set(words)) > 1 and len([tokens[i]] if tokens[i] not in word2pinyin else word2pinyin[tokens[i]]) > 1:
                # print(tokens[i], words, set(pinyins))
                polyphonic_entity_num += 1
                break
    return len(entities), polyphonic_entity_num


def get_dataset_ambiguity_entity_proportion(dataset, word2id_file, case_file, lexicon_window_size):
    print(dataset)
    word2id = json.load(open(word2id_file, encoding='utf8'))

    word2pinyin = json.load(open('word2pinyin.json', encoding='utf8'))
    sents, golds, preds = [[] for _ in range(3)]
    with open(case_file, encoding='utf8') as fin:
        for ith, line in enumerate(fin):
            if ith % 4 == 0:
                sents.append(line.strip())
            if ith % 4 == 1:
                golds.append(literal_eval(line.strip()))
            if ith % 4 == 2:
                preds.append(literal_eval(line.strip()))

    total_entities_num, polyphonic_ambiguity_entities_num = 0, 0
    for sent, gold, pred in zip(sents, golds, preds):
        entity_num, polyphonic_entity_num = lexicon_match(word2id, word2pinyin, sent, gold, lexicon_window_size)
        total_entities_num += entity_num
        polyphonic_ambiguity_entities_num += polyphonic_entity_num

    results = total_entities_num, polyphonic_ambiguity_entities_num, round(polyphonic_ambiguity_entities_num / total_entities_num * 100, 2)
    print(f"Total entities: {results[0]}; polyphonic_entities: {results[1]}; proportion: {results[2]}")
    return results


if __name__ == '__main__':
    import os
    lexicon_window_size_dict = {'msra': 9, 'ontonotes4': 9, 'resume': 13, 'weibo': 7}
    datasets = ['msra', 'ontonotes4', 'resume', 'weibo']
    for dataset in datasets:
        dir_path = r'D:\Documents\工作文档\SchoolWork\2020.9.1_AIPolicy实验\2021.5.10_数据集实验\SOTA'
        # print(dataset)
        case_study_file = os.path.join(dir_path, f'{dataset}_case.txt')
        word2id_file = f'../../input/benchmark/entity/{dataset}/word_freq/ctb/word_freq_w1-$.json'
        # parse_case_study_file(case_study_file)
        get_dataset_ambiguity_entity_proportion(dataset, word2id_file, case_study_file, lexicon_window_size_dict[dataset])
        # break