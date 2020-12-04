import sys
sys.path.append('..')
import pasaie
from pasaie import pasaner, pasare

import unittest

class TestInference(unittest.TestCase):

    def test_policy_bmoes_bert_crf(sef):
        entity_model = pasaie.pasaner.get_model('policy_bmoes/bert_lstm_crf0')
        # relation_model = pasaie.pasare.get_model('test-policy/bert_entity_dsp_dice_fgm_attention_test1')
        # relation_model = pasaie.pasare.get_model('test-policy/bert_entity_dsp_dice_fgm_attention_cat_test1')
        relation_model = pasaie.pasare.get_model('test-policy/bert_entity_dice_alpha0.6_fgm0')
        while True:
            text = input()
            tokens, entities = entity_model.infer(text)
            print(tokens)
            print(entities)
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    item = {'token':tokens, 'h': {'pos':entities[i][0], 'entity': entities[i][1]},
                            't': {'pos':entities[j][0], 'entity': entities[j][1]}}
                    relation_type, score = relation_model.infer(item)
                    if relation_type != 'Other':
                        print(f'{entities[i]} -> {entities[j]}: {relation_type}')


if __name__ == '__main__':
    # import os
    # print(os.path.abspath('./'))
    unittest.main()