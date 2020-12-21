import sys
sys.path.append('..')
import pasaie
from pasaie import pasaner, pasare, pasaap

import os
import six
import configparser
import unittest
from collections import OrderedDict

import torch


class TestInference(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super(TestInference, self).__init__(methodName)
        project_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(project_path, 'config.ini'))
        self.entity_model_path_list = ['policy_bmoes/bert_lstm_crf_micro_f1_0',
                                       'policy_bmoes/bert_lstm_mrc_dice_fgm_micro_f1_0',
                                       'policy_bmoes/bert_lstm_mrc_dice_autoweighted_fgm_micro_f1_0',
                                       'policy_bmoes/multi_bert_dice_fgm0',
                                       'policy_bmoes/single_bert_dice_fgm0']
        self.relation_model_path_list = ['test-policy/bert_entity_dice_fgm_ltp_dsp_attention_cat_micro_f1_0',
                                         'test-policy/bert_entity_dice_fgm_ddp_dsp_attention_cat_micro_f1_0']
        self.sentence_importance_path_list = ['sentence_importance_judgement/base_textcnn_ce_fgm1']
        self.test_file = '../input/benchmark/testdata/hc_test_corpus.txt'

    def test_policy_bmoes_bert_crf(self):
        query_path = os.path.join(self.config['path']['ner_dataset'], 'policy', 'query.txt')
        with open(query_path, 'r', encoding='utf-8') as qf:
            query_dict = dict(line.strip().split() for line in qf)
        entity_model_path = self.entity_model_path_list[2]
        relation_model_path = self.relation_model_path_list[1]
        entity_model = pasaie.pasaner.get_model(entity_model_path)
        relation_model = pasaie.pasare.get_model(relation_model_path)
        with open(self.test_file, 'r') as f, open('./result_1220.csv', 'w') as resf:
            resf.write('id,test,entities,relations\n')
            for i, line in enumerate(f):
                line = line.strip()
                if 'mrc' in entity_model_path:
                    tokens, entities = entity_model.infer(line, query_dict)
                else:
                    tokens, entities = entity_model.infer(line)
                print(tokens)
                d = OrderedDict()
                for entity in entities:
                    if (entity[0], entity[2]) in d:
                        d[(entity[0], entity[2])] += '/' + entity[1]
                    else:
                        d[(entity[0], entity[2])] = entity[1]
                entities = []
                for k, v in d.items():
                    entities.append((k[0], v, k[1]))
                entities = sorted(entities, key=lambda x: x[0])
                print(entities)
                res_line = str(i) + ',' + line + ',' + str(entities) + ','
                relations = set()
                for i in range(len(entities)):
                    for j in range(len(entities)):
                        if i == j:
                            continue
                        item = {'token': tokens, 'h': {'pos': entities[i][0], 'entity': entities[i][1]},
                                't': {'pos': entities[j][0], 'entity': entities[j][1]}}
                        relation_type, score = relation_model.infer(item)
                        if relation_type != 'Other':
                            relations.add((entities[i][2], relation_type, entities[j][2]))  # 基于mrc的实体抽取会出现实体对应多个类型
                for entity1, relation, entity2 in relations:
                    print(f'{entity1} -> {entity2}: {relation}')
                res_line += str(list(relations)) + '\n'
                resf.write(res_line)
                resf.flush()


if __name__ == '__main__':
    unittest.main()
