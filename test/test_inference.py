import sys
sys.path.append('..')
import pasaie
from pasaie import pasaner, pasare

import unittest

class TestInference(unittest.TestCase):

    def test_policy_bmoes_bert_crf(sef):
        entity_model = pasaie.pasaner.get_model('policy_bmoes_bert_lstm_crf')
        relation_model = pasaie.pasare.get_model('test-policy/bert_entity_dice_fgm0')
        while True:
            text = input()
            tokens, entities = entity_model.infer(text)
            print(tokens)
            print(entities)
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    item = {'token':tokens, 'h': {'pos':[entities[i][0], entities[i][0] + len(entities[i][2])], 'entity': entities[i][1]},
                            't': {'pos':[entities[j][0], entities[j][0] + len(entities[j][2])], 'entity': entities[j][1]}}
                    relation_type, score = relation_model.infer(item)
                    if relation_type != 'Other':
                        print(f'{entities[i]} -> {entities[j]}: {relation_type}')


if __name__ == '__main__':
    # import os
    # print(os.path.abspath('./'))
    unittest.main()