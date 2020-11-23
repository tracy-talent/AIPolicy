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
from pasaie.pasaap.tools import search_target_sentences, cut_sent, simple_sentence_filter, plot_tree
from pasaie.metrics import Mean

project_path = '/'.join(os.path.abspath(__file__).replace('\\', '/').split('/')[:-4])
config = configparser.ConfigParser()
config.read(os.path.join(project_path, 'config.ini'))
avg_sent_len = Mean()
max_sent_len = 0


class LogicNode:

    def __init__(self,
                 is_root,
                 logic_type,
                 sentence=None):
        """
            Construct a logic node.
        :param is_root: boolean, if is_root is True, then this node has some sub-nodes. Otherwise, this node only has its corresponding sentence.
        :param logic_type: str, can only be 'AND', 'OR', None
        :param sentence: str, the corresponding sentence of node
        """
        if is_root and logic_type is None:
            raise ValueError('The logic_type can not be None when it is a root-node.')
        if logic_type:
            logic_type = logic_type.upper()
            if logic_type not in ['AND', 'OR']:
                raise ValueError("The logic_type of a sentence-node can only be `AND` or `OR`: {}.".format(logic_type))

        self.logic_type = logic_type
        self.is_root = is_root
        self.sent = sentence
        self.children = {}

    def add_node(self, node, node_key=None):
        if node_key is None:
            node_key = 'item{}'.format(len(self.children) + 1)
        self.children[node_key] = node

    def convert_to_root_node(self, logic_type, sentence=None):
        """
            Transform a sentence-node to a root-node.
        :param logic_type: str, can only be 'AND', 'OR'.
        :param sentence: str or None.
        :return:
        """
        if logic_type is None:
            raise ValueError('logic_type can not be None.')
        if logic_type:
            logic_type = logic_type.upper()
            if logic_type not in ['AND', 'OR']:
                raise ValueError("The logic_type of a sentence-node can only be `AND` or `OR`: {}.".format(logic_type))
        self.logic_type = logic_type
        self.is_root = True
        self.sent = sentence
        self.children = {}

    def convert_to_nested_dict(self):
        """
            Transform the logic tree of a article to a nested dictionary.
        :return:
        """
        if not self.is_root:
            raise Exception("Nested dictionary should be converted from a root-node, not a sentence-node.")
        if self.sent:
            key = self.sent + '-' + self.logic_type
        else:
            key = self.logic_type
        nested_dict = {key: {}}
        sub_dict = nested_dict[key]
        for str_idx, node in self.children.items():
            if node.is_root:
                child_dict = node.convert_to_nested_dict()
                sub_dict[str_idx] = child_dict
            else:
                sub_dict[str_idx] = node.sent
        return nested_dict

    def clean_children(self):
        self.children = {}


class LogicTree:

    def __init__(self, root, json_path):
        """
            Construct a logic Tree.
        :param root: LogicNode or None.
        :param json_path: str or None.
        """
        if root:
            self.root = root
        elif json_path:
            self.root = self.construct_tree_from_json(json_path)
        else:
            raise ValueError("Variable `root` and `json_path` cant not both be None!")

        if json_path:
            self.name = json_path.rsplit('/', maxsplit=1)[-1].rsplit('\\', maxsplit=1)[-1].split('.')[0]
        else:
            self.name = 'tree'

    def save_as_json(self, output_path):
        """
            Save as json file.
        :param output_path: str, path of json file
        :return:
        """
        output_dir = output_path.rsplit('/', maxsplit=1)[0].rsplit('\\', maxsplit=1)[0]
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf8') as fout:
            nested_dict = self.root.convert_to_nested_dict()
            json.dump(nested_dict, fout, indent=1, ensure_ascii=False)

    def construct_tree_from_json(self, json_path):
        """
            Construct logic tree from json file.
        :param json_path: str
        :return:
        """
        with open(json_path, 'r', encoding='utf8') as fin:
            nested_dict = json.load(fin)
            if len(nested_dict.keys()) != 1:
                raise ValueError("The number of root key in nested dictionary must be one!")
            root_key = list(nested_dict.keys())[0]
            if '-' in root_key:
                logic_type = root_key.split('-')[-1]
                sentence = root_key.split('-')[0]
            else:
                logic_type = root_key.split('-')[0]
                sentence = None
            root_node = LogicNode(is_root=True, logic_type=logic_type, sentence=sentence)

            self._construct_tree_from_dict(list(nested_dict.values())[0], root_node)
            return root_node

    def _construct_tree_from_dict(self, nested_dict, parent_node):
        """
            Construct logic tree from nested dictionary.
        :param nested_dict: dict
        :param parent_node: LogicNode
        :return:
        """
        for key, value in nested_dict.items():
            if isinstance(value, dict):
                node_key = list(value.keys())[0]
                if '-' in node_key:
                    logic_type = node_key.split('-')[-1]
                    sentence = node_key.split('-')[0]
                else:
                    logic_type = node_key.split('-')[0]
                    sentence = None
                node = LogicNode(is_root=True, logic_type=logic_type, sentence=sentence)
                self._construct_tree_from_dict(value[node_key], node)
            else:
                node = LogicNode(is_root=False, logic_type=None, sentence=value)
            parent_node.add_node(node, key)

    def convert_to_nested_dict(self):
        """
            Convert tree to a nested dictionary.
        :return:
        """
        return self.root.convert_to_nested_dict()

    def save_as_png(self, output_dir, filename):
        plot_tree(self.convert_to_nested_dict(), output_dir=output_dir, name=filename)


def parse_corpus(corpus_dir,
                 output_dir,
                 is_rawtext=False):
    """
        Parse a article into a logic tree.
    :param corpus_dir: str, directory path of brat system corpus
    :param output_dir: str, output directory of parsing logic-tree
    :param is_rawtext: boolean, determine whether need to find target sentences or not
    :return:
    """
    os.makedirs(output_dir, exist_ok=True)
    files = [os.path.join(corpus_dir, file) for file in os.listdir(corpus_dir) if file.endswith('.txt')]
    miss_list = []
    target_sentences = {}
    for file in files:
        if is_rawtext:
            sentences = search_target_sentences(file)
        else:
            sentences = [line.strip() for line in open(file, 'r', encoding='utf8')]
        filename = file.rsplit('/', maxsplit=1)[-1].rsplit('\\', maxsplit=1)[-1]

        if len(sentences) == 0:
            miss_list.append(filename)
        else:
            target_sentences[filename] = sentences

    mismatched_dict = {"mismatched_list": miss_list}
    with open(os.path.join(output_dir, 'mismatched-files.json'), 'w', encoding='utf8') as fout:
        json.dump(mismatched_dict, fout, indent=1, ensure_ascii=False)

    target_dir = os.path.join(output_dir, "extracted-articles")
    if is_rawtext and os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    for filename, sentences in target_sentences.items():
        if is_rawtext:
            os.makedirs(target_dir, exist_ok=True)
            with open(os.path.join(target_dir, filename), 'w', encoding='utf8') as fout:
                for line in sentences:
                    fout.write(line.strip() + '\n')
        parse_article(output_dir, filename, sentences)
        # if filename == '3.txt':
        #     parse_article(output_dir, filename, sentences)

    with open(os.path.join(output_dir, 'sentence_statistic.json'), 'w', encoding='utf8') as fout:
        sent_dict = {'avg_sent_len': avg_sent_len.avg,
                     'max_sent_len': max_sent_len}
        json.dump(sent_dict, fout, indent=1, ensure_ascii=False)


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
    os.makedirs(os.path.join(output_dir, 'logic_tree'), exist_ok=True)
    article_tree.save_as_json(os.path.join(output_dir, 'logic_tree', filename.replace('.txt', '.json')))
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
    # parse_corpus(corpus_dir=os.path.join(output_path, 'extracted-articles'),
    #              output_dir=output_path,
    #              is_rawtext=False)


