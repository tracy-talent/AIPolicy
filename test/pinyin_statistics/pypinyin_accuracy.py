# -*-coding:utf-8-*-
"""
@Time    : 2021/7/12 20:31
@Author  : Mecthew
@File    : pypinyin_accuracy.py
"""
import json
from pypinyin import lazy_pinyin, Style
import numpy as np
from ast import literal_eval


def lexicon_match(texts, word2id, lexicon_window_size):
    """
    Args:
        tokens (list): token list
    Returns:
        indexed_bmes (list[list]): index of tokens' segmentation label in matched words.
        indexed lexicons (list[list]): encode ids of tokens' matched words.
        indexed pinyins (list[list]): encode ids of tokens' pinyin in matched words.
    """

    matched_words = set()
    for text in texts:
        tokens = list(text)
        for i in range(len(tokens)):
            for w in range(lexicon_window_size, 1, -1):
                for p in range(w):
                    if i - p < 0:
                        break
                    if i - p + w > len(tokens):
                        continue
                    word = ''.join(tokens[i - p:i - p + w])
                    if word in word2id and word not in matched_words:
                        matched_words.add(word)
    return matched_words


def get_nerfile_lexicon_pinyin_pairs(filepath, word2id, lexicon_window_size):
    texts = []
    with open(filepath, 'r', encoding='utf8') as fin:
        text = []
        for line in fin:
            if line.strip():
                text.append(line.strip().split()[0])
            else:
                texts.append(text)
                text = []
    matched_words = lexicon_match(texts, word2id, lexicon_window_size)
    return matched_words


def get_dataset_lexicon_pinyin_pairs(dataset):
    suffix = 'clip256.' if dataset in ['msra', 'ontonotes4'] else ''
    if dataset == 'msra':
        file_dict = {'train': f'../../input/benchmark/entity/{dataset}/train.char.{suffix}bmoes',
                     'test': f'../../input/benchmark/entity/{dataset}/test.char.{suffix}bmoes'}
    else:
        file_dict = {'train': f'../../input/benchmark/entity/{dataset}/train.char.{suffix}bmoes',
                     'dev': f'../../input/benchmark/entity/{dataset}/dev.char.{suffix}bmoes',
                     'test': f'../../input/benchmark/entity/{dataset}/test.char.{suffix}bmoes'}

    lexicon_window_size_dict = {'msra': 9, 'ontonotes4': 9, 'resume': 13, 'weibo': 7}
    word2id_path = f'../../input/benchmark/entity/{dataset}/word_freq/ctb/word_freq_w2-$.json'
    word2id = json.load(open(word2id_path, 'r', encoding='utf8'))
    lexicons = set()

    for k, filepath in file_dict.items():
        lexicons = lexicons.union(get_nerfile_lexicon_pinyin_pairs(filepath, word2id, lexicon_window_size_dict[dataset]))

    lexicon_pinyin_pairs = set()
    for lex in lexicons:
        pinyin = lazy_pinyin(lex, neutral_tone_with_five=True, style=Style.TONE3, strict=False)
        lexicon_pinyin_pairs.add((lex, tuple(pinyin)))
    lexicon_pinyin_pairs = list(lexicon_pinyin_pairs)
    with open(f'./{dataset}_lexicon_pinyin.json', 'w', encoding='utf8') as fout:
        json.dump(lexicon_pinyin_pairs, fout, indent=0, ensure_ascii=False)
    return lexicon_pinyin_pairs


def is_zh_word(word):
    word = word.lower()
    for ch in word:
        if not (0x4e00 <= ord(ch) <= 0x9fa5):
            return False
    else:
        return True


def random_select(dataset, select_num=100):
    lexicon_pinyin_pairs = json.load(open(f'./{dataset}_lexicon_pinyin.json', encoding='utf8'))
    lexicon_pinyin_pairs = [pairs for pairs in lexicon_pinyin_pairs if len(pairs[0]) >= 4 and is_zh_word(pairs[0])]
    length = len(lexicon_pinyin_pairs)
    indices = np.random.choice(range(length), select_num, replace=False)
    with open(f'./{dataset}_selected.json', 'w', encoding='utf8') as fout:
        select_samples = [lexicon_pinyin_pairs[idx] for idx in indices]
        json.dump(select_samples, fout, indent=0, ensure_ascii=False)


def parse_case_study_file(case_study_file):
    word2pinyin = json.load(open('word2pinyin.json', encoding='utf8'))
    golds = []
    preds = []
    with open(case_study_file, encoding='utf8') as fin:
        for ith, line in enumerate(fin):
            if ith % 4 == 1:
                golds.append(literal_eval(line.strip()))
            elif ith % 4 == 2:
                preds.append(literal_eval(line.strip()))

    correct = 0
    total_gold, total_preds = 0, 0
    for gold_list, pred_list in zip(golds, preds):
        correct += len(set(gold_list).intersection(set(pred_list)))
        total_gold += len(gold_list)
        total_preds += len(pred_list)

    recall = correct / total_gold
    prec = correct / total_preds
    f1 = 2 * recall * prec / (recall + prec)
    print("Total",  round(prec*100, 2), round(recall*100, 2), round(f1*100, 2))

    poly_golds, nonpoly_golds, poly_preds, nonpoly_preds = [[] for _ in range(4)]
    for gold_list, pred_list in zip(golds, preds):
        poly_gold_list, nonpoly_gold_list, poly_pred_list, nonpoly_pred_list = [[] for _ in range(4)]
        for entity in gold_list:
            entity_name = entity[-1]
            gold_poly_flag = False
            for zh_ch in entity_name:
                if zh_ch in word2pinyin and len(word2pinyin[zh_ch]) > 1:
                    poly_gold_list.append(entity)
                    gold_poly_flag = True
                    break
            if not gold_poly_flag:
                nonpoly_gold_list.append(entity)
        
        for entity in pred_list:
            entity_name = entity[-1]
            pred_poly_flag = False
            for zh_ch in entity_name:
                if zh_ch in word2pinyin and len(word2pinyin[zh_ch]) > 1:
                    poly_pred_list.append(entity)
                    pred_poly_flag = True
                    break
            if not pred_poly_flag:
                nonpoly_pred_list.append(entity)
        assert len(poly_gold_list) + len(nonpoly_gold_list) == len(gold_list)
        assert len(poly_pred_list) + len(nonpoly_pred_list) == len(pred_list)
        poly_golds.append(poly_gold_list)
        nonpoly_golds.append(nonpoly_gold_list)
        poly_preds.append(poly_pred_list)
        nonpoly_preds.append(nonpoly_pred_list)
    
    poly_correct = 0
    total_poly_gold, total_poly_pred = 0, 0
    for gold_list, pred_list in zip(poly_golds, poly_preds):
        poly_correct += len(set(gold_list).intersection(set(pred_list)))
        total_poly_gold += len(gold_list)
        total_poly_pred += len(pred_list)

    recall = poly_correct / total_poly_gold
    prec = poly_correct / total_poly_pred
    f1 = 2 * recall * prec / (recall + prec)
    print("Poly",  round(prec*100, 2), round(recall*100, 2), round(f1*100, 2))
    
    nonpoly_correct = 0
    total_nonpoly_gold, total_nonpoly_pred = 0, 0
    for gold_list, pred_list in zip(nonpoly_golds, nonpoly_preds):
        nonpoly_correct += len(set(gold_list).intersection(set(pred_list)))
        total_nonpoly_gold += len(gold_list)
        total_nonpoly_pred += len(pred_list)

    recall = nonpoly_correct / total_nonpoly_gold
    prec = nonpoly_correct / total_nonpoly_pred
    f1 = 2 * recall * prec / (recall + prec)
    print("Nonpoly",  round(prec*100, 2), round(recall*100, 2), round(f1*100, 2))


def sample_case_study_from_case_file(dataset, case_study_file, sample_num=200):
    word2pinyin = json.load(open('word2pinyin.json', encoding='utf8'))
    sents = []
    golds = []
    preds = []
    with open(case_study_file, encoding='utf8') as fin:
        for ith, line in enumerate(fin):
            if ith % 4 == 0:
                sents.append(line.strip())
            if ith % 4 == 1:
                golds.append(literal_eval(line.strip()))
            elif ith % 4 == 2:
                preds.append(literal_eval(line.strip()))

    new_sents, new_golds, new_preds = [[] for _ in range(3)]
    for sent, gold, pred in zip(sents, golds, preds):
        if len(gold) > 1:
            new_sents.append(sent)
            new_gold = []
            for entity in gold:
                entity_name = entity[-1]
                poly_flag = False
                for zh_ch in entity_name:
                    if zh_ch in word2pinyin and len(word2pinyin[zh_ch]) > 1:
                        poly_flag = True
                        break
                new_gold.append((*entity, int(poly_flag)))
            gold = new_gold
            new_golds.append(gold)
            new_preds.append(pred)

    sample_num = min(sample_num, len(new_sents))
    case_study_dir = case_study_file.replace('\\', '/').rsplit('/', maxsplit=1)[0]
    with open(f'{case_study_dir}/{dataset}_sample{sample_num}_case.txt', 'w', encoding='utf8') as fout:
        sample_indices = np.random.choice(np.arange(len(new_sents)), size=sample_num, replace=False)
        total_entities_num = 0
        for idx in sample_indices:
            total_entities_num += len(new_golds[idx])
        fout.write(f'Total entities num: {total_entities_num}\n')
        for idx in sample_indices:
            fout.write(f'{new_sents[idx]}\n{new_golds[idx]}\n{new_preds[idx]}\n\n')


if __name__ == '__main__':
    # for dataset in ['msra', 'ontonotes4', 'weibo', 'resume']:
        # get_dataset_lexicon_pinyin_pairs(dataset)
        # print(f"finish {dataset}")
        # random_select(dataset, select_num=100)
    import os
    datasets = ['msra', 'ontonotes4', 'resume', 'weibo']
    for dataset in datasets:
        dir_path = r'D:\Documents\工作文档\SchoolWork\2020.9.1_AIPolicy实验\2021.5.10_数据集实验\SOTA'
        print(dataset)
        case_study_file = os.path.join(dir_path, f'{dataset}_case.txt')
        # parse_case_study_file(case_study_file)
        sample_case_study_from_case_file(dataset, case_study_file)