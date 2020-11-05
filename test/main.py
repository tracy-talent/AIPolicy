import os
import sys
from collections import deque
from ast import literal_eval

sys.path.append('..')
import pasaie
from pasaie import pasaner, pasare, pasaap
import configparser

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
        pasaie.pasaap.framework.parse_corpus(corpus_dir=corpus_path,
                                             output_dir=output_path,
                                             is_rawtext=True)

    os.makedirs(os.path.join(output_path, 'final-result'), exist_ok=True)
    for file in os.listdir(os.path.join(output_path, 'logic_tree')):
        if not file.endswith('.json'):
            continue
        jsonpath = os.path.join(output_path, 'logic_tree', file)
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
                        item = {'token': tokens, 'h': {'pos': [entities[i][0], entities[i][0] + len(entities[i][2])]},
                                't': {'pos': [entities[j][0], entities[j][0] + len(entities[j][2])]}}
                        reversed_item = {'token': tokens,
                                         't': {'pos': [entities[i][0], entities[i][0] + len(entities[i][2])]},
                                         'h': {'pos': [entities[j][0], entities[j][0] + len(entities[j][2])]}}
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
                            node=pasaie.pasaap.framework.LogicNode(is_root=False, logic_type=None, sentence=str(relation_pairs)),
                            node_key='relations'
                        )

        tree.save_as_json(output_path=os.path.join(output_path, 'final-result', file))


def fine_gained_parse(text, entity_list, relation_list):
    """
        Further fine-gained parsing for input sentence according to entities and relations.
    :param text: str, input sentence
    :param entity_list: list of tuple, entity tuple is like (start_pos, entity_type, entity_name)
    :param relation_list: list of tuple, each tuple is like (head_entity_tuple, tail_entity_tuple, relation_type)
    :return:
        list of LogicNode.
    """
    split_separators = [',', ';']
    logic_symbols = {'AND': ['和', '与', '且', '、'], 'OR': ['或']}
    entities_in_relations = set()
    for head, tail, rel_type in relation_list:
        entities_in_relations.add(head)
        entities_in_relations.add(tail)
    entities_not_in_relations = set(entity_list) - set(entities_in_relations)

    begin_idx = 0
    text_spans = []
    for idx, ch in enumerate(text):
        if ch in split_separators:
            text_spans.append((begin_idx, text[begin_idx: idx]))
            begin_idx = idx + 1
    if begin_idx < len(text):
        text_spans.append((begin_idx, text[begin_idx:]))

    node_list = []
    for begin_idx, span in text_spans:
        for entity in entities_not_in_relations:
            entity_start_pos = entity[0]
            if begin_idx <= entity_start_pos < begin_idx + len(span):
                node_list.append(pasaie.pasaap.framework.LogicNode(is_root=False, logic_type=None, sentence=entity[2]))
    for relation in relation_list:
        head, tail, rel_type = relation
        rel_term = (head[2], tail[2], rel_type)
        node_list.append(pasaie.pasaap.framework.LogicNode(is_root=False, logic_type=None, sentence=str(rel_term)))
    return node_list


def main_entry(dataset):
    entity_model = pasaie.pasaner.get_model('policy_bmoes_bert_crf')
    relation_model = pasaie.pasare.get_model('test-policy_bert_entity')
    # entity_model = None
    # relation_model = None
    corpus_path = os.path.join(config['path']['input'], 'benchmark', 'article_parsing', dataset)
    output_path = os.path.join(config['path']['output'], 'article_parsing', dataset)

    pipeline(corpus_path=corpus_path,
             output_path=output_path,
             is_rawtext=False,
             entity_model=entity_model,
             relation_model=relation_model)


def main_entry2(dataset):
    corpus_path = os.path.join(config['path']['input'], 'benchmark', 'article_parsing', dataset)
    output_path = os.path.join(config['path']['output'], 'article_parsing', dataset)

    os.makedirs(os.path.join(output_path, 'final-result2'), exist_ok=True)
    for file in os.listdir(os.path.join(output_path, 'final-result')):
        if not file.endswith('.json'):
            continue
        jsonpath = os.path.join(output_path, 'final-result', file)
        tree = pasaie.pasaap.framework.LogicTree(root=None, json_path=jsonpath)
        queue = deque()
        queue.append((None, tree.root))
        while queue:
            parent, node = queue.popleft()
            if node.is_root:
                for name, child_node in node.children.items():
                    queue.append((node, child_node))
            else:
                if 'entities' in parent.children.keys():
                    entity_list = literal_eval(parent.children['entities'].sent)
                    relation_list = [] if 'relations' not in parent.children else literal_eval(parent.children['relations'].sent)
                    node_list = fine_gained_parse(text=parent.sent.rsplit('-', maxsplit=1)[0],
                                                  entity_list=entity_list,
                                                  relation_list=relation_list)
                    parent.clean_children()
                    for sub_node in node_list:
                        parent.add_node(sub_node)

        tree.save_as_json(output_path=os.path.join(output_path, 'final-result2', file))
        tree.save_as_png(output_dir=os.path.join(output_path, 'final-png'), filename=file)


if __name__ == '__main__':
    main_entry2('raw-policy')
