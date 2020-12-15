import sys

sys.path.append('..')
import pasaie
from pasaie import pasaner, pasare, pasaap
import os
import six
import configparser

import unittest


class TestInference(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super(TestInference, self).__init__(methodName)
        project_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(project_path, 'config.ini'))
        self.entity_model_path_list = ['policy_bmoes/bert_lstm_crf0',
                                       'policy_bmoes/bert_lstm_mrc_dice_fgm0',
                                       'policy_bmoes/multi_bert_dice_fgm0',
                                       'policy_bmoes/single_bert_dice_fgm0']
        self.relation_model_path_list = ['test-policy/bert_entity_dsp_dice_fgm_attention_cat_test0',
                                         'test-policy/bert_entity_dice_alpha0.6_fgm0']
        self.sentence_importance_path_list = ['sentence_importance_judgement/base_textcnn_ce_fgm1']

    def test_policy_bmoes_bert_crf(self):
        query_path = os.path.join(self.config['path']['ner_dataset'], 'policy', 'query.txt')
        with open(query_path, 'r', encoding='utf-8') as qf:
            query_dict = dict(line.strip().split() for line in qf)
        entity_model_path = self.entity_model_path_list[3]
        relation_model_path = self.relation_model_path_list[1]
        sentence_importance_path = self.sentence_importance_path_list[0]
        # entity_model = pasaie.pasaner.get_model(entity_model_path[0])
        # relation_model = pasaie.pasare.get_model(relation_model_path[1])
        entity_model = pasaie.pasaner.get_model(entity_model_path)
        relation_model = pasaie.pasare.get_model(relation_model_path)
        sentence_importance_model = pasaie.pasaap.get_model(model_path=sentence_importance_path)
        while True:
            if six.PY3:
                text = input().encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
            elif six.PY2:
                text = input().decode('utf-8', errors='ignore').encode('utf-8', errors='ignore')
            print("text is importance? {}".format(sentence_importance_model.infer(text)))
            if 'mrc' in entity_model_path:
                tokens, entities = entity_model.infer(text, query_dict)
            else:
                tokens, entities = entity_model.infer(text)
            print(tokens)
            print(entities)
            relations = set()
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    item = {'token': tokens, 'h': {'pos': entities[i][0], 'entity': entities[i][1]},
                            't': {'pos': entities[j][0], 'entity': entities[j][1]}}
                    relation_type, score = relation_model.infer(item)
                    if relation_type != 'Other':
                        relations.add((entities[i], relation_type, entities[j]))  # 基于mrc的实体抽取会出现实体对应多个类型
            for entity1, relation, entity2 in relations:
                print(f'{entity1} -> {entity2}: {relation}')


if __name__ == '__main__':
    unittest.main()
