import sys
sys.path.append('..')
import pasaie
from pasaie import pasaner, pasare, pasaap

import os
from functools import cmp_to_key
import six
import configparser
from collections import defaultdict

import torch
import pandas as pd


class Node(object):
    def __init__(self, edge, entity_pos, entity_name, entity_type=None):
        self.entity_pos = entity_pos
        self.entity_name = entity_name
        self.entity_type = entity_type
        self.edge = edge
        self.accessed = False


class Edge(object):
    def __init__(self, rel_name, node):
        self.rel_name = rel_name
        self.node = node
        self.accessed = False


class CorpusInference(object):

    def __init__(self):
        super(CorpusInference, self).__init__()
        self.rel2str = {'Lt': '小于', 'Lte': '小于等于', 'Gt': '大于', 'Gte': '大于等于', 'Eq': '等于', 'Is': '是',
                        'Proportion': '占比', 'Locate': '位于', 'Has': '有', 'Engage': '从事', 'Meet': '满足', 'Joint': ''}

        project_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(project_path, 'config.ini'))
        self.entity_model_path_list = ['policy_bmoes/bert_lstm_crf_micro_f1_0',
                                       'policy_bmoes/bert_lstm_mrc_dice_fgm_micro_f1_0',
                                       'policy_bmoes/bert_lstm_mrc_dice_autoweighted_fgm_micro_f1_0',
                                       'policy_bmoes/multi_bert_dice_fgm0',
                                       'policy_bmoes/single_bert_dice_fgm0']
        self.relation_model_path_list = ['test-policy/bert_entity_context_dsp_tail_bert_ddp_dsp_attention_context_dice_fgm_dpr0.5_micro_f1_0',
                                         'test-policy/bert_entity_context_dsp_tail_bert_ltp_dsp_attention_context_dice_fgm_dpr0.5_micro_f1_0']
        self.sentence_importance_path_list = ['sentence_importance_judgement/base_textcnn_ce_fgm1']
        self.test_file = '../input/benchmark/testdata/hc_test_corpus.txt'
        with open('../input/benchmark/relation/test-policy/permitted_entity_pair.txt', 'r', encoding='utf-8') as f:
            self.permitted_entity_pair = eval(f.readline().strip())


    def filter_entity_overlap(self, entities: list) -> list:
        filter_entities = []
        for ent1 in entities:
            flag = True
            for ent2 in entities:
                if ent1 == ent2:
                    continue
                if ent1[0][0] >= ent2[0][0] and ent1[0][1] <= ent2[0][1]:
                    flag = False
                    break
            if flag:
                filter_entities.append(ent1)
        return filter_entities


    def dfs(self, graph, ent, ent_range, req_str, policy_requirements, depth):
        req_str += ent[1]
        ent_range.append(ent[0])
        if ent not in graph:
            if depth > 0:
                policy_requirements.append((set(ent_range), req_str))
                ent_range.pop()
            return
        flag = False
        for rel_ent in graph[ent]:
            if not rel_ent[2]:
                flag = True
                rel_ent[2] = True
                self.dfs(graph, rel_ent[1], ent_range, req_str + self.rel2str[rel_ent[0]], policy_requirements, depth + 1)
                rel_ent[2] = False
        if not flag and depth > 0:
            policy_requirements.append((set(ent_range), req_str))
        ent_range.pop()
            

    def generate_requirements(self, relations: list) -> list:
        relations = sorted(relations, key=cmp_to_key(lambda x, y: x[0][0] < y[0][0] or (x[0][0] == y[0][0] and x[2][0] < y[2][0])))
        du = defaultdict(lambda: 0)
        graph = defaultdict(list)
        for rel in relations:
            graph[rel[0]].append(list(rel[1:] + (False,)))
            du[rel[2]] += 1
        policy_requirements = []
        for ent in graph:
            if du[ent] == 0:
                self.dfs(graph, ent, [], '', policy_requirements, 0)
        filter_policy_requirements = []
        # 过滤长要求中的子要求
        for i, pri in enumerate(policy_requirements):
            flag = True
            for j, prj in enumerate(policy_requirements):
                if i == j:
                    continue
                if pri[0] & prj[0] == pri[0]:
                    flag = False
                    break
            if flag:
                filter_policy_requirements.append(pri[1])
        return filter_policy_requirements


    def infer(self):
        query_path = os.path.join(self.config['path']['ner_dataset'], 'policy', 'query.txt')
        with open(query_path, 'r', encoding='utf-8') as qf:
            query_dict = dict(line.strip().split() for line in qf)
        entity_model_path = self.entity_model_path_list[2]
        relation_model_path = self.relation_model_path_list[1]
        entity_model = pasaie.pasaner.get_model(entity_model_path)
        relation_model = pasaie.pasare.get_model(relation_model_path)
        with open(self.test_file, 'r') as f, open('./result_1223.csv', 'w') as csvf:
            df_data = {'text': [], 'entities': [], 'relations': [], 'requirements': []}
            csvf.write('text,entities,relations\n')
            for i, line in enumerate(f):
                line = line.strip()
                if 'mrc' in entity_model_path:
                    tokens, entities = entity_model.infer(line, query_dict)
                else:
                    tokens, entities = entity_model.infer(line)
                print(tokens)

                # extract entities
                d = {}
                for entity in entities:
                    if (entity[0], entity[2]) in d:
                        d[(entity[0], entity[2])] += '/' + entity[1]
                    else:
                        d[(entity[0], entity[2])] = entity[1]
                entities = []
                for k, v in d.items():
                    entities.append((k[0], v, k[1]))
                entities = self.filter_entity_overlap(entities)
                entities = sorted(entities, key=lambda x: x[0])
                print(entities)
                csv_line = line + ',' + "\"" + str(entities) + '\"'
                # extract relations
                relations = set()
                for i in range(len(entities)):
                    for j in range(len(entities)):
                        if i == j:
                            continue
                        if 'mrc' not in entity_model_path and (entities[i][1], entities[j][1]) not in self.permitted_entity_pair:
                            continue
                        # rule out entity pair has overlap
                        if max(entities[i][0][0], entities[j][0][0]) < min(entities[i][0][1], entities[j][0][1]):
                            continue
                        item = {'token': tokens, 'h': {'pos': entities[i][0], 'entity': entities[i][1]},
                                't': {'pos': entities[j][0], 'entity': entities[j][1]}}
                        relation_type, score = relation_model.infer(item)
                        if relation_type != 'Other':
                            relations.add(((entities[i][0], entities[i][2]), relation_type, (entities[j][0], entities[j][2])))  # 基于mrc的实体抽取会出现实体对应多个类型
                for entity1, relation, entity2 in relations:
                    print(f'{entity1} -> {entity2}: {relation}')
                relations = list(relations)
                csv_line += ",\"" + str(relations) + '\"' 

                # generate policy requirements
                policy_requirements = self.generate_requirements(relations)
                print('\n'.join(policy_requirements) + '\n')
                csv_line += ",\"" + str(policy_requirements) + '\"\n'

                # save result
                csvf.write(csv_line)
                csvf.flush()
                df_data['text'].append(line)
                df_data['entities'].append(entities)
                df_data['relations'].append(relations)
                df_data['requirements'].append(policy_requirements)
            df = pd.DataFrame(df_data)
            df.to_csv('./result_df_1223.csv', index=False)



if __name__ == '__main__':
    inferer = CorpusInference()
    inferer.infer()
