"""
 Author: liujian 
 Date: 2020-10-25 14:34:12 
 Last Modified by: liujian 
 Last Modified time: 2020-10-25 14:34:12 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer

import re
import os
import json
import unicodedata
from contextlib import ExitStack
from collections import defaultdict
import copy

from ..tokenization.utils import is_control, is_whitespace


def adjust_brat_ann(data_path):
    """adjust brat annotation, make entities occur before relation

    Args:
        data_path (str): brat annotation data path
    """
    filelist = os.listdir(data_path)
    ann = []
    txt = []
    for i in range(len(filelist)):
        if i % 2 == 0:
            ann.append(filelist[i])
        else:
            txt.append(filelist[i])
    for af in ann:
        with open(os.path.join(data_path, af), 'r', encoding='utf-8') as f1, \
            open(os.path.join(data_path, af + '.tmp'), 'w', encoding='utf-8') as f2:
            e = []
            r = []
            for line in f1:
                if line[0] == 'R':
                    r.append(line)
                else:
                    e.append(line)
            for x in e:
                f2.write(x)
            for x in r:
                f2.write(x)
            os.remove(os.path.join(data_path, af))
            os.rename(os.path.join(data_path, af + '.tmp'), os.path.join(data_path, af))

def get_brat_txt_ann(data_path):
    """get brat txt and ann file name under brat annotation data path

    Args:
        data_path (str): brat annotation data path

    Returns:
        filelist (list[tuple]): [(txt files), (ann files)]
    """
    filelist = os.listdir(data_path)
    file_name = set()
    for item in filelist:
        file_name.add(item[:-4])
    return [(file_predix + '.txt', file_predix + '.ann') for file_predix in file_name]


def get_entity_sentence(text, hpos, tpos):
    """ get [hpos, tpos) of sentence which contains entity by '\n'

    Args:
        text (str): relation text
        hpos (int): start position of relation text
        tpos (int): end position of relation text

    Returns:
        hpos (int): start position of relation text cut by '\n'
        tpos (int): end position of relation text cut by '\n'
    """
    text_len = len(text)
#     while hpos >= 0 and text[hpos] != '\n' and unicodedata.category(text[hpos]) != 'Po':
    while hpos >= 0 and text[hpos] != '\n':
        hpos -= 1
    if text[hpos] == '\n':
        hpos += 1
    while tpos < text_len and text[tpos] != '\n':
        tpos += 1
    return hpos, tpos


def get_relation_sentence(text, hpos, tpos):
    """ get [hpos, tpos) of sentence which contains relation by puctuation(，。？！)

    Args:
        text (str): relation text
        hpos (int): start position of relation text
        tpos (int): end position of relation text

    Returns:
        hpos (int): start position of relation text cut by punctuation
        tpos (int): end position of relation text cut by punctuation
    """
    text_len = len(text)
    punc = ',，;；：:。.!！?？'
    while hpos >= 0 and text[hpos] not in punc:
        hpos -= 1
    if text[hpos] in punc:
        hpos += 1
    while tpos < text_len and text[tpos] not in punc:
        tpos += 1
    return hpos, tpos


def clean(seqs, bios):
    """clean token seqs

    Args:
        seqs (list[str]): token list
        bios (list[str]): bio tag list
    """
    i = 0
    while i < len(seqs):
        seqs[i] = seqs[i].lower()
        if len(seqs[i]) > 1:
            i += 1
            continue
        cp = ord(seqs[i])
        cat = unicodedata.category(seqs[i])
        if seqs[i] == '\xa0' or cp == 0 or cp == 0xfffd or cat == 'Mn' or is_control(seqs[i]) or is_whitespace(seqs[i]):
            seqs.pop(i)
            bios.pop(i)
        else:
            i += 1


def get_bio(seqs, entities):
    """get seqs's bio tags from entities span

    Args:
        seqs (list[str]): token list
        entities (dict): entities dict which contain their span

    Returns:
        bios (list[str]): bio tag list
    """
    bios = ['O'] * len(seqs)
    for d in entities:
        for i in range(d['relpos'][0], d['relpos'][1]):
            if i == d['relpos'][0]:
                bios[i] = 'B-{}'.format(d['entity'])
            else:
                bios[i] = 'I-{}'.format(d['entity'])
    return bios


def merge_digit(seqs, bios):
    """merge digit as a whole

    Args:
        seqs (list[str]): token list
        bios (list[str]): bio tag list

    Returns:
        mseqs (list[str]): token list after digit merge
        mbios (list[str]): bio tag list after digit merge
    """
    mseqs = []
    mbios = []
    f = True
    for i in range(len(seqs)):
        if seqs[i] >= '0' and seqs[i] <= '9':
            if f:
                mseqs.append(seqs[i])
                mbios.append(bios[i])
                f = False
            else:
                mseqs[-1] += seqs[i]
        else:
            mseqs.append(seqs[i])
            mbios.append(bios[i])
            f = True
    return mseqs, mbios


def brat_parse(data_path, reltag_directed=False, brat_ann_adjusted=True):
    """parse brat annotation, get content, entity and relation information

    Args:
        data_path (str): brat annotation data path
        reltag_directed (bool, optional): whether add direction in reltag. Defaults to False.
        brat_ann_adjusted (bool, optional): whether have adjusted brat .ann file. Defaults to True.

    Returns:
        corpus_content (list[str]): text content of every article
        corpus_entity (list[list[dict]]): entity list of every article
        corpus_relation (list[list[dict]]): relation list of every article
    """
    corpus_content = []
    corpus_entity = []
    corpus_relation = []
    if not brat_ann_adjusted:
        adjust_brat_ann(data_path)
    filelist = get_brat_txt_ann(data_path)
    for tf, af in filelist:
        with open(os.path.join(data_path, tf), 'r', encoding='utf-8') as f:
            corpus_content.append(f.read())
        with open(os.path.join(data_path, af), 'r', encoding='utf-8') as f:
            entity = []
            relation = []
            entity2id = {}
            for line in f:
                items = line.strip().split()
                if line[0] == 'T' and items[1] != 'rTips':
                    entity2id[items[0]] = len(entity2id)
                    hpos, tpos = get_entity_sentence(corpus_content[-1], int(items[2]), int(items[3]))
                    d = {'article':tf, # article file name
                        'sentence': (hpos, tpos), # sentence span index, entity located in this sentence
                        'name':items[4], # entity name, ideally article content in [hpos, tpos) should be items[4]
                        'pos': (int(items[2]), int(items[3])), # entity absolute position index in article
                        'entity':items[1]} # entity type
                    d['relpos'] = (d['pos'][0] - hpos, d['pos'][1] - hpos) # left open and right close range[), entity relative position index in article
                    entity.append(d)
                elif line[0] == 'R':
                    hid, tid = 2, 3
                    if entity[entity2id[items[2][5:]]]['pos'][0] > entity[entity2id[items[3][5:]]]['pos'][0]:
                        hid, tid = 3, 2
                    if entity[entity2id[items[2][5:]]]['sentence'] != entity[entity2id[items[3][5:]]]['sentence']:
                        print(tf, line) # detect relation which across multiple lines
                        hpos, tpos = get_entity_sentence(corpus_content[-1], 
                                                        entity[entity2id[items[hid][5:]]]['pos'][0], 
                                                        entity[entity2id[items[tid][5:]]]['pos'][1])
                    else:
                        hpos, tpos = entity[entity2id[items[2][5:]]]['sentence']
                    if reltag_directed:
                        d = {'article':tf, # article file name
                            'sentence': (hpos, tpos), # sentence span index, relation located in this sentence
                            'h':entity[entity2id[items[hid][5:]]], # head entity
                            't':entity[entity2id[items[tid][5:]]], # tail entity
                            'relation':items[1] + ('(e1,e2)' if hid < tid else '(e2,e1)')} # relation type
                    else:
                        d = {'article':tf, 
                            'sentence': (hpos, tpos), 
                            'h':entity[entity2id[items[2][5:]]], 
                            't':entity[entity2id[items[3][5:]]], 
                            'relation':items[1]}
                    d['h']['relpos'] = (d['h']['pos'][0] - hpos, d['h']['pos'][1] - hpos) # adjust head entity relpos
                    d['t']['relpos'] = (d['t']['pos'][0] - hpos, d['t']['pos'][1] - hpos) # adjust head entity relpos
                    relation.append(d)
            corpus_entity.append(entity)
            corpus_relation.append(relation)
    return corpus_content, corpus_entity, corpus_relation


def get_entity_type_cluster(corpus_entity):
    """get entities of every entity type

    Args:
        corpus_entity (list[list[dict]]): entity list of every article

    Returns:
        entity_type_cluster (dict[str:list]): map from entity type to relative entities
    """
    entity_type_cluster = defaultdict(set)
    for i, x in enumerate(corpus_entity):
        if len(x) == 0:
            continue
        for d in x:
            entity_cnt += 1        
            entity_type_cluster[d['entity']].add(d['name'])
    for k, v in entity_type_cluster.items():
        entity_type_cluster[k] = list(v)
    return entity_type_cluster


def get_relation_type_cluster(corpus_relation):
    """"get entity pairs of every relation type

    Args:
        corpus_relation (list[list[dict]]): relation list of every article

    Returns:
        relation_type_cluster (dict[str:list]): map from relation type to relative entity pairs
    """
    relation_type_cluster = defaultdict(set)
    for i, x in enumerate(corpus_relation):
        if len(x) == 0:
            continue
        for d in x:
            relation_type_cluster[d['relation']].add((d['h']['name'], d['t']['name']))
    for k, v in relation_type_cluster:
        relation_type_cluster[k] = list(v)
    return relation_type_cluster


def get_sentence_entity_cluster(corpus_entity):
    """get entities of every sentence, and entities is ordered by their position is sentence

    Args:
        corpus_entity (list[list[dict]]): entity list of every article

    Returns:
        sent_entity_cluster (list[dict[str:list]]): map from sentence span to relative entities of every article
    """
    sent_entity_cluster = []
    for i, x in enumerate(corpus_entity):
        if len(x) == 0:
            sent_entity_cluster.append([])
            continue
        entity_cluster = defaultdict(list)
        for d in x:
            entity_cluster[d['sentence']].append(copy.deepcopy(d))
        for entity in entity_cluster:
            entity_cluster[entity].sort(key=lambda y: y['pos'][0])
        sent_entity_cluster.append(entity_cluster)
    return sent_entity_cluster


def get_sentence_relation_cluster(corpus_relation):
    """get relations of every sentence

    Args:
        corpus_relation (list[list[dict]]): relation list of every article

    Returns:
        sent_relation_cluster [list[dict[str:list]]]: map from sentence span to relative relations of every article
    """
    sent_relation_cluster = []
    for i, x in enumerate(corpus_relation):
        if len(x) == 0:
            sent_relation_cluster.append([])
            continue
        relation_cluster = defaultdict(list)
        for d in x:
            relation_cluster[d['sentence']].append(copy.deepcopy(d))
        sent_relation_cluster.append(relation_cluster)
    return sent_relation_cluster


def get_entity_pos_of_relation(bios):
    """get entity pair position of relation

    Args:
        bios (list[str]): bio tag sequence

    Returns:
        span1 (tuple): left entity span
        span2 (tuple): right entity span
    """
    f = False
    i = 0
    while i < len(bios):
        if bios[i][0] == 'B':
            spos = i
            while i + 1 < len(bios) and bios[i + 1][0] == 'I':
                i += 1
            if not f:
                span1 = (spos, i + 1)
                f = True
            else:
                span2 = (spos, i + 1)
        i += 1
    return span1, span2

    
def entity_verify(corpus_content, corpus_entity):
    """entity verification, detect if the entity is consistent with it's absolute position in article

    Args:
        corpus_content (list[str]): text content of every article
        corpus_entity (list[list[dict]]): entity list of every article

    Returns:
        error_entities (list): list of incorrect entity information
    """
    error_entities = []
    for i, x in enumerate(corpus_entity):
        if len(x) == 0:
            continue
        for d in x:
            if corpus_content[i][d['pos'][0]:d['pos'][1]] != d['name']:
                error_entities.append((corpus_content[i][d['pos'][0]:d['pos'][1]], d))
    return error_entities


def save_custom_dict(corpus_entity, file_path):
    """construct custom dict from entity name,  abandon entity contains digits

    Args:
        corpus_entity (list[list[dict]]): entity list of every article
        file_path (str): file save path of custom dict
    """
    custom_dict = set()
    for i, x in enumerate(corpus_entity):
        if len(x) == 0:
            continue
        for d in x:
            if re.search('\d+', d['name']) is None:
                custom_dict.add(d['name'])
    with open(file_path, 'w', encoding='utf-8') as f:
        for word in custom_dict:
            f.write(word + '\n')


def save_corpus_char(corpus_content, corpus_entity, data_path):
    """save corpus entity to 'data_path/corpus.char', format:
    article_name ('token', 'tag') ('token', 'tag') ...  

    Args:
        corpus_content (list[str]): text content of every article
        corpus_entity (list[list[dict]]): entity list of every article
        data_path (str): file save path of corpus.char
    """
    with open(os.path.join(data_path, 'corpus.char'), 'w', encoding='utf-8') as f:
        for i, x in enumerate(corpus_entity):
            if len(x) == 0:
                continue
            sent_cluster = defaultdict(list)
            for d in x:
                sent_cluster[d['sentence']].append(d)
            for k, v in sent_cluster.items():
                seqs = list(corpus_content[i][k[0]:k[1]])
                bios = get_bio(seqs, v)
                clean(seqs, bios)
                seqs, bios = merge_digit(seqs, bios)
                f.write(v[0]['article'] + ' ' + ' '.join([str(t) for t in list(zip(seqs, bios))]) + '\n')


def load_corpus_char(corpus_file_path):
    """load corpus.char file

    Args:
        corpus_file_path (str): corpus file path

    Returns:
        seqs (list[list]): token list
        bios (list[list]): tag list
    """
    corpus_seqs = []
    corpus_bios = []
    with open(corpus_file_path, 'r', encoding='utf-8') as f:
        for lid, line in enumerate(f):
            line = line.strip()
            seq_bio = []
            f = False
            f1 = False
            for i in range(len(line)):
                if line[i] == '(' and not f:
                    s = i
                    f = True
                elif line[i] == ')' and f1:
                    seq_bio.append(eval(line[s:i+1]))
                    f = False
                    f1 = False
                elif line[i] == ',' and line[i:i+2] == ', ':
                    f1 = True
            seqs, bios = [list(t) for t in zip(*seq_bio)]
            corpus_seqs.append(seqs)
            corpus_bios.append(bios)
    return corpus_seqs, corpus_bios


def construct_entity_corpus_from_corpus_char(corpus_file_path, train_test_data_path):
    """train test split from corpus char, save them to train_test_data_path(every line is 'token tag')

    Args:
        corpus_file_path (str): corpus file path
        train_test_data_path (str): data save path of train and test
    """
    with ExitStack() as stack:
        tagset = set()
        train_test_file_name = ['train.char.bio', 'test.char.bio', 'tag2id.bio']
        train_test_file = [stack.enter_context(open(os.path.join(train_test_data_path, fname), 'w', encoding='utf-8')) for fname in train_test_file_name]
        pre_article_name = ''
        train_size = len(open(corpus_file_path, encoding='utf-8').readlines()) // 5 * 4
        with open(corpus_file_path, 'r', encoding='utf-8') as f:
            flag = False
            for lid, line in enumerate(f):
                line = line.strip()
                seq_bio = []
                f = False
                f1 = False
                article_name = line[:line.find(' ')]
                for i in range(len(line)):
                    if line[i] == '(' and not f:
                        s = i
                        f = True
                    elif line[i] == ')' and f1:
                        seq_bio.append(eval(line[s:i+1]))
                        f = False
                        f1 = False
                    elif line[i] == ',' and line[i:i+2] == ', ':
                        f1 = True
                seqs, bios = [list(t) for t in zip(*seq_bio)]
                if not flag and lid >= train_size and pre_article_name != article_name:
                    flag = True
                for token, tag in zip(seqs, bios):
                    train_test_file[flag].write(token + ' ' + tag + '\n')
                    tagset.add(tag)
                train_test_file[flag].write('\n')
                pre_article_name = article_name
        tagset.remove('O')
        attrset = set()
        for tag in tagset:
            attrset.add(tag[2:])
        for attr in attrset:
            if ('B-' + attr) in tagset:
                train_test_file[2].write(f'B-{attr}\n')
            if ('I-' + attr) in tagset:
                train_test_file[2].write(f'I-{attr}\n')
        train_test_file[2].write('O')


def construct_bert_entity_corpus_from_corpus_char(corpus_file_path, train_test_data_path, tokenizer=None):
    """train test split(train:test=4:1) from corpus char, save them to train_test_data_path(every line is 'token tag')

    Args:
        corpus_file_path (str): corpus file path
        train_test_data_path (str): data save path of train and test
        tokenizer (function): tokenizer for word segmentation, example: jieba.lcut
    """
    with ExitStack() as stack:
        tagset = set()
        train_test_file_name = ['train.bert.char.bio', 'test.bert.char.bio', 'tag2id.bio']
        train_test_file = [stack.enter_context(open(os.path.join(train_test_data_path, fname), 'w', encoding='utf-8')) for fname in train_test_file_name]
        pre_article_name = ''
        train_size = len(open(corpus_file_path, encoding='utf-8').readlines()) // 5 * 4
        with open(corpus_file_path, 'r', encoding='utf-8') as f:
            flag = False
            for lid, line in enumerate(f):
                line = line.strip()
                seq_bio = []
                f = False
                f1 = False
                article_name = line[:line.find(' ')]
                for i in range(len(line)):
                    if line[i] == '(' and not f:
                        s = i
                        f = True
                    elif line[i] == ')' and f1:
                        seq_bio.append(eval(line[s:i+1]))
                        f = False
                        f1 = False
                    elif line[i] == ',' and line[i:i+2] == ', ':
                        f1 = True
                seqs, bios = [list(t) for t in zip(*seq_bio)]
                clean(seqs, bios)
                
                if tokenizer is not None:
                    tokens = tokenizer(''.join(seqs)) # example: tokenizer = jieba.lcut
                    # 根据jieba分词进行bert vocab构造
                    pos = 0
                    token = ''
                    for i in range(len(tokens)):
                        if re.fullmatch('\d+', tokens[i]) and i < len(tokens) - 1 and re.fullmatch('\d+', tokens[i + 1]):
                            token += tokens[i]
                            continue
                        token += tokens[i]
                        t = 0
                        while t < len(token):
                            sl = len(seqs[pos])
                            if t != 0:
                                seqs[pos] = '##' + seqs[pos]
                            t += sl
                            pos += 1
                        token = ''

                # 把实体作为整体bert vocab 构造
                i = 0
                while i < len(bios):
                    if bios[i].startswith('B-'):
                        j = i + 1
                        while j < len(bios) and bios[j].startswith('I-'):
                            j += 1
                        if seqs[i].startswith('##'):
                            seqs[i] = seqs[i][2:]
                        for p in range(i + 1, j):
                            if not seqs[p].startswith('##'):
                                seqs[p] = '##' + seqs[p]
                        i = j
                    else:
                        i += 1 
                if not flag and lid >= train_size and pre_article_name != article_name:
                    flag = True
                for token, tag in zip(seqs, bios):
                    train_test_file[flag].write(token + ' ' + tag + '\n')
                    tagset.add(tag)
                train_test_file[flag].write('\n')
                pre_article_name = article_name
        tagset.remove('O')
        attrset = set()
        for tag in tagset:
            attrset.add(tag[2:])
        for attr in attrset:
            if ('B-' + attr) in tagset:
                train_test_file[2].write(f'B-{attr}\n')
            if ('I-' + attr) in tagset:
                train_test_file[2].write(f'I-{attr}\n')
        train_test_file[2].write('O')


def construct_relation_corpus_from_brat(data_path, out_path, reltag_directed=False, brat_ann_adjusted=True, max_seq_len=256):
    """construct relation corpus from brat annotation, saved as rel2id.json, train.txt, val.txt(train:val=4:1)

    Args:
        data_path (str): brat annotation 
        out_path (str): output path of relation corpus
        reltag_directed (bool, optional): whether add direction in reltag. Defaults to False.
        brat_ann_adjusted (bool, optional): whether have adjusted brat .ann file. Defaults to True.
        max_seq_len (int, optional): max length of relation span. Defaults to 256.
    """
    corpus_content, corpus_entity, corpus_relation = brat_parse(data_path, reltag_directed=reltag_directed, brat_ann_adjusted=brat_ann_adjusted)
    sent_entity_cluster = get_sentence_entity_cluster(corpus_entity)
    sent_relation_cluster = get_sentence_relation_cluster(corpus_relation)
    cnt = 0 # 非Other关系计数
    all_relation_corpus = []
    relation_type_stats = defaultdict(lambda: 0)
    relation_type_cluster = defaultdict(list)
    too_long_relation = []
    for i in range(len(sent_relation_cluster)):
        if len(sent_relation_cluster[i]) == 0:
            continue
        for rk, rv in sent_relation_cluster[i].items():
            for ek, ev in sent_entity_cluster[i].items():
                if rk != ek:
                    continue
                for j in range(len(ev)):
                    for k in range(j + 1 if reltag_directed else 0, len(ev)):
                        if j == k:
                            continue
                        flag = False # Other关系为False，其它为True
                        for et in rv:
                            if et['h'] == ev[j] and et['t'] == ev[k]:
                                t = copy.deepcopy(et)
                                if ev[j]['pos'][0] < ev[k]['pos'][0]:
                                    hpos, tpos = get_relation_sentence(corpus_content[i], ev[j]['pos'][0], ev[k]['pos'][1])
                                else:
                                    hpos, tpos = get_relation_sentence(corpus_content[i], ev[k]['pos'][0], ev[j]['pos'][1])
                                t['h']['relpos'] = (t['h']['pos'][0] - hpos, t['h']['pos'][1] - hpos)
                                t['t']['relpos'] = (t['t']['pos'][0] - hpos, t['t']['pos'][1] - hpos)
                                seqs = list(corpus_content[i][hpos:tpos])
                                bios = get_bio(seqs, [t['h'], t['t']])
                                clean(seqs, bios)
                                seqs, bios = merge_digit(seqs, bios)
                                if t['h']['pos'][0] < t['t']['pos'][0]:
                                    t['h']['pos'], t['t']['pos'] = get_entity_pos_of_relation(bios)
                                else:
                                    t['t']['pos'], t['h']['pos'] = get_entity_pos_of_relation(bios)
                                t['token'] = seqs
                                for attr in ['article', 'relpos', 'sentence']:
                                    del t['h'][attr]
                                    del t['t'][attr]
                                relation_type_stats[t['relation']] += 1
                                cnt += 1
                                rel_dict = {'article':t['article'], 
                                                'sentence':t['sentence'], 
                                                'token':t['token'], 
                                                'h':t['h'], 't':t['t'], 
                                                'relation':t['relation']}
                                if len(seqs) <= max_seq_len - 6: # 考虑bert_entity模型加了6个token，而且bert tokenizer后的序列长度也未知
                                    all_relation_corpus.append(rel_dict)
                                    relation_type_cluster[t['relation']].append(rel_dict)
                                else:
                                    too_long_relation.append(rel_dict)
                                flag = True
                                break
                        if not flag:
                            if ev[j]['pos'][0] < ev[k]['pos'][0]:
                                hpos, tpos = get_relation_sentence(corpus_content[i], ev[j]['pos'][0], ev[k]['pos'][1])
                            else:
                                hpos, tpos = get_relation_sentence(corpus_content[i], ev[k]['pos'][0], ev[j]['pos'][1])
                            h = copy.deepcopy(ev[j])
                            t = copy.deepcopy(ev[k])
                            h['relpos'] = (h['pos'][0] - hpos, h['pos'][1] - hpos)
                            t['relpos'] = (t['pos'][0] - hpos, t['pos'][1] - hpos)
                            seqs = list(corpus_content[i][hpos:tpos])
                            bios = get_bio(seqs, [h, t])
                            clean(seqs, bios)
                            seqs, bios = merge_digit(seqs, bios)
                            if h['pos'][0] < t['pos'][0]:
                                h['pos'], t['pos'] = get_entity_pos_of_relation(bios)
                            else:
                                t['pos'], h['pos'] = get_entity_pos_of_relation(bios)
                            for attr in ['article', 'relpos', 'sentence']:
                                del h[attr]
                                del t[attr]
                            relation_type_stats['Other'] += 1
                            rel_dict = {'article':rv[0]['article'], 
                                        'sentence':(hpos, tpos), 
                                        'token':seqs, 
                                        'h':h, 't':t, 
                                        'relation':'Other'}
                            if len(seqs) <= max_seq_len - 6:
                                all_relation_corpus.append(rel_dict)
                                relation_type_cluster['Other'].append(rel_dict)
                            else:
                                too_long_relation.append(rel_dict)
    print(f'there are {len(too_long_relation)} relations exceed {max_seq_len} length!')
    rel2id = {}
    for rel in relation_type_stats:
        rel2id[rel] = len(rel2id)
    with open(os.path.join(out_path, 'policy_rel2id.json'), 'w', encoding='utf-8') as f:
        json.dump(rel2id, f, ensure_ascii=False)
    with open(os.path.join(out_path, 'train.txt'), 'w', encoding='utf-8') as tf, \
        open(os.path.join(out_path, 'val.txt'), 'w', encoding='utf-8') as vf:
        for k, v in relation_type_cluster.items():
            for i, item in enumerate(v):
                json.dump(item, vf if i < len(v) / 5 else tf, ensure_ascii=False)
                vf.write('\n') if i < len(v) / 5 else tf.write('\n')


def construct_english_bert_ner_corpus_from_bmoes(input_file, output_file, pretrain_path):
    tokenizer = BertTokenizer.from_pretrained(pretrain_path)
    with open(input_file, 'r', encoding='utf-8') as rf, open(output_file, 'w', encoding='utf-8') as wf:
        seqs = []
        for line in rf:
            line = line.strip().split()
            if len(line) > 0:
                seqs.append(line)
            elif len(seqs) > 0:
                tokens, tags = list(zip(*seqs))
                bert_tokens = tokenizer.tokenize(' '.join(tokens))
                bert_tags = []
                tpos, tlen = 0, 0
                bpos, blen = 0, 0
                while tpos < len(tokens):
                    if tags[tpos][0] == 'B' or tags[tpos][0] == 'S':
                        if tags[tpos][0] == 'B':
                            tlen = 0
                            while tags[tpos][0] != 'E':
                                tlen += len(tokens[tpos])
                                tpos += 1
                        else:
                            tlen = len(tokens[tpos])
                        attr = tags[tpos][2:]
                        blen = 0
                        last_bpos = bpos
                        while blen < tlen:
                            if bert_tokens[bpos].startswith('##'):
                                blen += len(bert_tokens[bpos][2:])
                            else:
                                blen += len(bert_tokens[bpos])
                            bpos += 1
                        if last_bpos + 1 == bpos:
                            bert_tags.append(f'S-{attr}')
                        else:
                            bert_tags.append(f'B-{attr}')
                            for _ in range(last_bpos + 1, bpos - 1):
                                bert_tags.append(f'M-{attr}')
                            bert_tags.append(f'E-{attr}')
                    else:
                        tlen = len(tokens[tpos])
                        blen = 0
                        last_bpos = bpos
                        while blen < tlen:
                            if bert_tokens[bpos].startswith('##'):
                                blen += len(bert_tokens[bpos][2:])
                            else:
                                blen += len(bert_tokens[bpos])
                            bpos += 1
                        for j in range(last_bpos, bpos):
                            bert_tags.append('O')
                    tpos += 1
                for token_tag_pair in zip(bert_tokens, bert_tags):
                    wf.write(' '.join(token_tag_pair) + '\n')
                wf.write('\n')
                seqs = []

                
def paint_relation_hist(corpus_relation, data_path):
    """paint relation type histogram

    Args:
        corpus_relation (list[list[dict]]): entity list of every article
        data_path (str): file save path of relation type histogram
    """
    relation_type_stats = defaultdict(lambda: 0)
    for rel in corpus_relation:
        if len(rel) == 0:
            continue
        for d in rel:
            relation_type_stats[d['relation']] += 1
    json.dump(relation_type_stats, open(os.path.join(data_path, 'relation_type_stats.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    plt.figure()
    plt.xticks(range(len(relation_type_stats)), relation_type_stats.keys(), rotation=90)
    plt.bar(x=range(len(relation_type_stats)), height=relation_type_stats.values())
    plt.savefig(os.path.join(data_path, 'relation_hist.jpg'), bbox_inches='tight')
    # plt.show()


def paint_entity_hist(corpus_entity, data_path):
    """paint entity type histogram

    Args:
        corpus_entity (list[list[dict]]): 
        data_path (str): file save path of entity type histogram
    """
    entity_type_stats = defaultdict(lambda: 0)
    for ent in corpus_entity:
        if len(ent) == 0:
            continue
        for d in ent:
            entity_type_stats[d['entity']] += 1
    json.dump(entity_type_stats, open(os.path.join(data_path, 'entity_type_stats.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    plt.figure()
    plt.xticks(range(len(entity_type_stats)), entity_type_stats.keys(), rotation=90)
    plt.bar(x=range(len(entity_type_stats)), height=entity_type_stats.values())
    plt.savefig(os.path.join(data_path, 'entity_hist.jpg'), bbox_inches='tight')
    # plt.show()
                        