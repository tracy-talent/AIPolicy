import os
import sys
import re
from collections import deque
from ast import literal_eval
import configparser

sys.path.append('..')
import pasaie
from pasaie import pasaner, pasare, pasaap
from pasaie.pasaap.tools import LogicNode, convert_json_to_png, judge_sent_logic_type

project_path = '/'.join(os.path.abspath(__file__).replace('\\', '/').split('/')[:-2])
config = configparser.ConfigParser()
config.read(os.path.join(project_path, 'config.ini'))


def pipeline(corpus_path,
             output_path,
             is_rawtext,
             entity_model,
             relation_model
             ):
    if is_rawtext:
        pasaie.pasaap.framework.parse_corpus(corpus_dir=corpus_path)

    os.makedirs(os.path.join(output_path, 'sentence_level_logic_tree'), exist_ok=True)
    for file in os.listdir(os.path.join(output_path, 'pruning_tree')):
        if not file.endswith('.json'):
            continue
        jsonpath = os.path.join(output_path, 'pruning_tree', file)
        tree = pasaie.pasaap.framework.LogicTree(root=None, json_path=jsonpath)

        queue = deque()
        queue.append(tree.root)
        while queue:
            node = queue.popleft()
            if node.is_root:
                for name, child_node in node.children.items():
                    queue.append(child_node)
            else:
                sentence = node.sent
                tokens, entities = entity_model.infer(sentence)
                # relation_pairs = [entity[2] for entity in entities]
                relation_pairs = []
                for i in range(len(entities)):
                    for j in range(i + 1, len(entities)):
                        item = {'token': tokens,
                                'h': {'pos': [entities[i][0], entities[i][0] + len(entities[i][2])],
                                      'entity': entities[i][1]},
                                't': {'pos': [entities[j][0], entities[j][0] + len(entities[j][2])],
                                      'entity': entities[j][1]}}
                        reversed_item = {'token': tokens,
                                         't': {'pos': [entities[i][0], entities[i][0] + len(entities[i][2])],
                                               'entity': entities[i][1]},
                                         'h': {'pos': [entities[j][0], entities[j][0] + len(entities[j][2])],
                                               'entity': entities[j][1]}}
                        relation_type, score = relation_model.infer(item)
                        if relation_type not in ['Other', 'other']:
                            relation_pairs.append((entities[i], entities[j], relation_type))
                        else:
                            relation_type, score = relation_model.infer(reversed_item)
                            if relation_type not in ['Other', 'other']:
                                relation_pairs.append((entities[j], entities[i], relation_type))

                if entities:
                    node.convert_to_root_node(logic_type='AND', sentence=sentence)
                    node.add_node(
                        node=pasaie.pasaap.framework.LogicNode(is_root=False, logic_type=None, sentence=str(entities)),
                        node_key='entities'
                    )
                    if relation_pairs:
                        node.add_node(
                            node=pasaie.pasaap.framework.LogicNode(is_root=False, logic_type=None,
                                                                   sentence=str(relation_pairs)),
                            node_key='relations'
                        )

        tree.save_as_json(output_path=os.path.join(output_path, 'sentence_level_logic_tree', file))


def fine_gained_parse(tokens, entity_list, relation_list):
    """
        Further fine-gained parsing for input sentence according to entities and relations.
    :param text: str, input sentence
    :param entity_list: list of tuple, entity tuple is like (start_pos, entity_type, entity_name)
    :param relation_list: list of tuple, each tuple is like (head_entity_tuple, tail_entity_tuple, relation_type)
    :return:
        list of LogicNode.
    """
    entities_in_relations = set()
    for head, tail, rel_type in relation_list:
        entities_in_relations.add(head)
        entities_in_relations.add(tail)
    entities_not_in_relations = list(set(entity_list) - set(entities_in_relations))
    entity_list = entities_not_in_relations

    if ':' in ''.join(tokens)[:20] or '：' in ''.join(tokens)[:20]:
        # if tokens.count(':') + tokens.count("：") > 1:
        #     print(f"{''.join(tokens)} has more than 1 :")
        summary_end_idx = 0
        for sidx in range(len(tokens)):
            if tokens[sidx] in [':', '：']:
                summary_end_idx = sidx + 1
                break
        summary_sent = tokens[:summary_end_idx]
        summary_logic_type = judge_sent_logic_type(''.join(summary_sent))
    else:
        summary_sent = ''
        summary_logic_type = 'AND'

    or_separators = ['；', ';']
    text_begin_idx = len(summary_sent)  # begin idx
    short_sentence_indices = []
    for idx in range(len(tokens)):
        if tokens[idx] in or_separators:
            short_sentence_indices.append((text_begin_idx, idx + 1))
            text_begin_idx = idx + 1
    if text_begin_idx < len(tokens):
        short_sentence_indices.append((text_begin_idx, len(tokens)))

    sub_node_list = []
    for idx_tuple in short_sentence_indices:
        begin_idx, end_idx = idx_tuple
        tmp_entity_list, tmp_relation_list = [], []
        for entity in entity_list:
            pos_b, cate, ent_name = entity
            if begin_idx <= pos_b < end_idx:
                tmp_entity_list.append((pos_b - begin_idx, cate, ent_name))
        for relation in relation_list:
            head, tail, rel_type = relation
            t_pos_b, t_cate, t_ent_name = tail
            if begin_idx <= t_pos_b < end_idx:
                tmp_tail = (t_pos_b - begin_idx, t_cate, t_ent_name)
                tmp_relation_list.append((head, tmp_tail, rel_type))
        if tmp_entity_list or tmp_relation_list:
            sub_node = parse_short_sentence_with_and(tokens[begin_idx: end_idx], tmp_entity_list, tmp_relation_list)
            sub_node_list.append(sub_node)

    if len(sub_node_list) > 1:
        root_node = LogicNode(is_root=True, logic_type='OR', sentence=None)
        for sub_node in sub_node_list:
            root_node.add_node(sub_node)
    elif len(sub_node_list) == 1:
        root_node = sub_node_list[0]
        if summary_logic_type == 'OR':
            root_node.logic_type = summary_logic_type
    else:
        root_node = None

    return root_node


def parse_short_sentence_with_and(short_sentence, entity_list, relation_list):
    and_separators = [',', '，']
    sentence_begin_idx = 0
    short_sentence_indices = []
    for idx in range(len(short_sentence)):
        if short_sentence[idx] in and_separators:
            short_sentence_indices.append((sentence_begin_idx, idx + 1))
            sentence_begin_idx = idx + 1
    if sentence_begin_idx < len(short_sentence):
        short_sentence_indices.append((sentence_begin_idx, len(short_sentence)))

    sub_node_list = []
    for idx_tuple in short_sentence_indices:
        begin_idx, end_idx = idx_tuple
        tmp_entity_list, tmp_relation_list = [], []
        for entity in entity_list:
            pos_b, cate, ent_name = entity
            if begin_idx <= pos_b < end_idx:
                tmp_entity_list.append((pos_b - begin_idx, cate, ent_name))
        for relation in relation_list:
            head, tail, rel_type = relation
            t_pos_b, t_cate, t_ent_name = tail
            if begin_idx <= t_pos_b < end_idx:
                tmp_tail = (t_pos_b - begin_idx, t_cate, t_ent_name)
                tmp_relation_list.append((head, tmp_tail, rel_type))
        if tmp_entity_list or tmp_relation_list:
            sub_node = parse_span_with_or(short_sentence[begin_idx: end_idx], tmp_entity_list,
                                          tmp_relation_list)
            sub_node_list.append(sub_node)
    if len(sub_node_list) > 1:
        root_node = LogicNode(is_root=True, logic_type='AND', sentence=None)
        for sub_node in sub_node_list:
            root_node.add_node(sub_node)
    elif len(sub_node_list) == 1:
        root_node = sub_node_list[0]
    else:
        root_node = None
    return root_node


def parse_span_with_or(a_span, entity_list, relation_list):
    or_separators = ['、']
    span_begin_idx = 0
    short_sentence_indices = []
    for idx in range(len(a_span)):
        if a_span[idx] in or_separators:
            short_sentence_indices.append((span_begin_idx, idx + 1))
            span_begin_idx = idx + 1
    if span_begin_idx < len(a_span):
        short_sentence_indices.append((span_begin_idx, len(a_span)))
    entity_list.sort(key=lambda x: x[0])
    relation_list.sort(key=lambda x: x[1][0])

    short_sentence_span = [[] for _ in range(len(short_sentence_indices))]
    for entity in entity_list:
        pos_b, cate, ent_name = entity
        for ith, idx_tuple in enumerate(short_sentence_indices):
            if idx_tuple[0] <= pos_b < idx_tuple[1]:
                short_sentence_span[ith].append(f"{cate}: {ent_name}")
                # short_sentence_span[ith].append(str(entity))
                break

    for relation in relation_list:
        head, tail, rel_type = relation
        # TODO: relation仅根据尾实体位置确定位于哪个span
        t_pos_b, t_cate, t_ent_name = tail
        for ith, idx_tuple in enumerate(short_sentence_indices):
            if idx_tuple[0] <= t_pos_b < idx_tuple[1]:
                short_sentence_span[ith].append(f"({head[1]}:{head[2]}, {t_cate}:{t_ent_name}, {rel_type})")
                # short_sentence_span[ith].append(str(relation))
                break

    short_sentence_span = [span for span in short_sentence_span if len(span) > 0]
    sub_node_list = []
    for span in short_sentence_span:
        if len(span) > 1:
            sub_root = LogicNode(is_root=True, logic_type='AND', sentence=None)
            for ent_rel in span:
                sub_root.add_node(LogicNode(is_root=False, logic_type=None, sentence=ent_rel))
        else:
            sub_root = LogicNode(is_root=False, logic_type=None, sentence=span[0])
        sub_node_list.append(sub_root)
    if len(sub_node_list) > 1:
        root_node = LogicNode(is_root=True, logic_type='OR', sentence=None)
        for sub_node in sub_node_list:
            root_node.add_node(sub_node)
    else:
        root_node = sub_node_list[0]
    return root_node


def remove_node_without_entities(root_node):
    if root_node.is_root:
        del_names = []
        for child_name, child_node in root_node.children.items():
            if remove_node_without_entities(child_node):
                del_names.append(child_name)
        for name in del_names:
            del root_node.children[name]
        return len(root_node.children) == 0
    else:
        if 'entities' in root_node.parent.children or 'relations' in root_node.parent.children:
            # print(root_node.parent.sent)
            return False
        else:
            return True


def main_entry_step1(dataset, entity_model, relation_model):
    corpus_path = os.path.join(config['path']['input'], 'benchmark', 'article_parsing', dataset)
    output_path = os.path.join(config['path']['output'], 'article_parsing', dataset)

    is_rawtext = not os.path.exists(os.path.join(output_path, 'pruning_tree'))
    pipeline(corpus_path=corpus_path,
             output_path=output_path,
             is_rawtext=is_rawtext,
             entity_model=entity_model,
             relation_model=relation_model)


def main_entry_step2(dataset, entity_model):
    output_path = os.path.join(config['path']['output'], 'article_parsing', dataset)
    os.makedirs(os.path.join(output_path, 'element_level_logic_tree'), exist_ok=True)

    tokenizer = entity_model.sequence_encoder.tokenizer
    for file in os.listdir(os.path.join(output_path, 'sentence_level_logic_tree')):
        if not file.endswith('.json'):
            continue
        jsonpath = os.path.join(output_path, 'sentence_level_logic_tree', file)
        tree = pasaie.pasaap.framework.LogicTree(root=None, json_path=jsonpath)
        remove_node_without_entities(tree.root)
        queue = deque()
        if tree.root:
            queue.append((None, tree.root))
        while queue:
            parent, node = queue.popleft()
            if node.is_root:
                for name, child_node in node.children.items():
                    queue.append((node, child_node))
            else:
                if 'entities' in parent.children.keys():
                    entity_list = literal_eval(parent.children['entities'].get_sentence())
                    relation_list = [] if 'relations' not in parent.children \
                        else literal_eval(parent.children['relations'].get_sentence())
                    sub_node = fine_gained_parse(tokens=tokenizer.tokenize(parent.get_sentence()),
                                                 entity_list=entity_list,
                                                 relation_list=relation_list)
                    parent.clean_children()  # 这里正好把'entities' 和 'relations'节点调用两次parent的副作用消除
                    if sub_node:
                        parent.add_node(sub_node)
                    else:
                        print(f"{file}: cannot find {entity_list} in {parent.get_sentence()}")

        tree.save_as_json(output_path=os.path.join(output_path, 'element_level_logic_tree', file))
        try:
            tree.save_as_png(output_dir=os.path.join(output_path, 'element_level_logic_tree'), filename=file)
        except:
            pass


def main_entry():
    dataset = 'raw-policy'
    extracted_path = os.path.join(config['path']['output'], 'article_parsing', dataset, 'sentence_level_logic_tree')
    my_entity_model = pasaie.pasaner.get_model('policy_bmoes_bert_lstm_crf')
    my_relation_model = pasaie.pasare.get_model('test-policy/bert_entity_dice_fgm0')
    if not os.path.exists(extracted_path):
        print("Executing main_entry...")
        main_entry_step1(dataset, my_entity_model, my_relation_model)
    main_entry_step2(dataset, my_entity_model)


if __name__ == '__main__':
    main_entry()
    # convert_json_to_png(config['path']['output'] + '/article_parsing/raw-policy/element_level_logic_tree')
