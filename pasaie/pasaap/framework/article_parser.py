#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2020/10/26 15:57
# @Author:  Mecthew
import configparser
import json
import os
import shutil
import sys

sys.path.append('../../..')
from pasaie.pasaap.tools import get_target_tree, cut_sent, simple_sentence_filter, LogicTree, LogicNode
from pasaie.metrics import Mean

project_path = '/'.join(os.path.abspath(__file__).replace('\\', '/').split('/')[:-4])
config = configparser.ConfigParser()
config.read(os.path.join(project_path, 'config.ini'))
avg_sent_len = Mean()
max_sent_len = 0


def parse_corpus(corpus_dir):
    """
        Parse a article into a logic tree.
    :param corpus_dir: str, directory path of brat system corpus
    :param output_dir: str, output directory of parsing logic-tree
    :param is_rawtext: boolean, determine whether need to find target sentences or not
    :return:
    """
    files = [os.path.join(corpus_dir, file) for file in os.listdir(corpus_dir) if file.endswith('.txt')]
    for file in files:
        get_target_tree(file)
    # miss_list = []
    # target_sentences = {}
    # for file in files:
    #     if is_rawtext:
    #         sentences = get_target_sentences(file)
    #     else:
    #         sentences = [line.strip() for line in open(file, 'r', encoding='utf8')]
    #     filename = file.rsplit('/', maxsplit=1)[-1].rsplit('\\', maxsplit=1)[-1]
    #
    #     if len(sentences) == 0:
    #         miss_list.append(filename)
    #     else:
    #         target_sentences[filename] = sentences
    #
    # mismatched_dict = {"mismatched_list": miss_list}
    # with open(os.path.join(output_dir, 'mismatched_files.json'), 'w', encoding='utf8') as fout:
    #     json.dump(mismatched_dict, fout, indent=1, ensure_ascii=False)
    #
    # target_dir = os.path.join(output_dir, "target_sentences")
    # if is_rawtext and os.path.exists(target_dir):
    #     shutil.rmtree(target_dir)
    # for filename, sentences in target_sentences.items():
    #     if is_rawtext:
    #         os.makedirs(target_dir, exist_ok=True)
    #         with open(os.path.join(target_dir, filename), 'w', encoding='utf8') as fout:
    #             for line in sentences:
    #                 fout.write(line.strip() + '\n')
    #     parse_article(output_dir, filename, sentences)
    #     # if filename == '3.txt':
    #     #     parse_article(output_dir, filename, sentences)
    #
    # with open(os.path.join(output_dir, 'sentence_statistic.json'), 'w', encoding='utf8') as fout:
    #     sent_dict = {'avg_sent_len': avg_sent_len.avg,
    #                  'max_sent_len': max_sent_len}
    #     json.dump(sent_dict, fout, indent=1, ensure_ascii=False)


def parse_article(output_dir, filename, input_sentences):
    if input_sentences[-1] == '\n':
        del input_sentences[-1]

    spans = []
    span_idx = []
    for i in range(len(input_sentences)):
        if input_sentences[i].strip() == '':
            span_idx.append(i)
    span_idx.append(len(input_sentences))
    begin_idx = 0
    for cnt, idx in enumerate(span_idx, 1):
        spans.append(input_sentences[begin_idx: idx])
        begin_idx = idx + 1

    root_node = LogicNode(is_root=True, logic_type='And', sentence='文章')

    span_i = 1
    for span in spans:
        node = parse_span(input_span=span)
        if node:
            root_node.add_node(node, "span{}".format(span_i))
            span_i += 1

    article_tree = LogicTree(root=root_node, json_path=None)
    os.makedirs(os.path.join(output_dir, 'raw_text_logic_tree'), exist_ok=True)
    article_tree.save_as_json(os.path.join(output_dir, 'raw_text_logic_tree', filename.replace('.txt', '.json')))
    # plot_tree(article_tree.convert_to_nested_dict())


def parse_span(input_span):
    if len(input_span) <= 1:
        return None
    chinese_numbers = list('一二三四五六七八九十')
    arabic_numbers = list(str(i) for i in range(10))
    possible_combinations = [num + '.' for num in chinese_numbers + arabic_numbers] + \
                            ['（{}）'.format(num) for num in chinese_numbers + arabic_numbers]

    title = input_span[0]
    OR_expression =['下列条件中的一项', "条件之一"]
    if any(s in title for s in OR_expression):
        span_node = LogicNode(is_root=True, logic_type='OR', sentence='子段')
    else:
        span_node = LogicNode(is_root=True, logic_type='AND', sentence='子段')

    if all(not (s in input_span[1][:10]) for s in possible_combinations) and len(input_span[1]) < 20:
        subtitle = input_span[1]
        begin_idx = 2
        if "条件之一" in subtitle:
            span_node = LogicNode(is_root=True, logic_type='OR', sentence='子段')
        else:
            span_node = LogicNode(is_root=True, logic_type='AND', sentence='子段')
    else:
        begin_idx = 1

    sent_i = 1
    for sentence in input_span[begin_idx:]:
        sen_node = parse_sentence(sentence)
        if sen_node:
            span_node.add_node(sen_node, "sent{}".format(sent_i))
            sent_i += 1

    if len(span_node.children) == 0:
        return None
    else:
        return span_node


def parse_sentence(input_sentence, filter_func=simple_sentence_filter):
    global avg_sent_len, max_sent_len
    sub_sentences = cut_sent(input_sentence)
    sent_node = LogicNode(is_root=True, logic_type='AND', sentence='句子')

    sub_sent_i = 1
    for sub_sent in sub_sentences:
        sub_sent = sub_sent.strip()
        if sub_sent and filter_func(sub_sent):

            # sentence length statistic
            avg_sent_len.update(val=len(sub_sent), n=1)
            if len(sub_sent) > max_sent_len:
                max_sent_len = len(sub_sent)

            sub_node = LogicNode(is_root=False, logic_type=None, sentence=sub_sent)
            sent_node.add_node(sub_node, "subsent{}".format(sub_sent_i))
            sub_sent_i += 1
    if len(sent_node.children) == 0:
        return None
    else:
        return sent_node


if __name__ == '__main__':
    input_path = os.path.join(config['path']['input'], 'benchmark', 'article_parsing')
    output_path = os.path.join(config['path']['output'], 'article_parsing', 'raw-policy')
    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    parse_corpus(corpus_dir=os.path.join(input_path, 'raw-policy'),
                 output_dir=output_path,
                 is_rawtext=True)
    # parse_corpus(corpus_dir=os.path.join(output_path, 'target_sentences'),
    #              output_dir=output_path,
    #              is_rawtext=False)


