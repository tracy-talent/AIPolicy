import configparser

import os
import json

import sys
sys.path.append('..')
from pasaie.utils.dependency_parse import LTP_Parse, DDP_Parse, Stanza_Parse
from pasaie.utils.dependency_parse import construct_corpus_dsp
from pasaie.utils.timedec import timeit
from transformers import BertTokenizer

@timeit
def dsp(dsp_tool='ddp'):
    pretrain_path = '/home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext'
    tokenizer = BertTokenizer.from_pretrained(pretrain_path)
    num_added_tokens = tokenizer.add_tokens(['“', '”', '—'])
    path = '../input/benchmark/relation/test-policy'
    files = ['train', 'test', 'val']
    if 'dsp_tool' == 'stanza':
        dsp = Stanza_Parse()
    else:
        dsp = eval(f'{dsp_tool.upper()}_Parse')()
    max_len, min_len = 0, 1000000
    for fname in files:
        with open(os.path.join(path, f'test-policy_{fname}.txt'), 'r', encoding='utf-8') as f, open(os.path.join(path, f'test-policy_{fname}_tail_bert_{dsp_tool}_dsp_path.txt'), 'w', encoding='utf-8') as f2:
            for i, line in enumerate(f.readlines()):
                line = line.rstrip()
                if len(line) > 0:
                    line = eval(line)

                    sentence = line['token']
                    pos_head = line['h']['pos']
                    pos_tail = line['t']['pos']
                    pos_min = pos_head
                    pos_max = pos_tail
                    if pos_head[0] > pos_tail[0]:
                        pos_min = pos_tail
                        pos_max = pos_head
                        rev = True
                    else:
                        rev = False
                    sent0 = tokenizer.tokenize(''.join(sentence[:pos_min[0]]))
                    ent0 = tokenizer.tokenize(''.join(sentence[pos_min[0]:pos_min[1]]))
                    sent1 = tokenizer.tokenize(''.join(sentence[pos_min[1]:pos_max[0]]))
                    ent1 = tokenizer.tokenize(''.join(sentence[pos_max[0]:pos_max[1]]))
                    sent2 = tokenizer.tokenize(''.join(sentence[pos_max[1]:]))
                    pos1_1 = len(sent0) if not rev else len(sent0 + ent0 + sent1)
                    pos1_2 = pos1_1 + len(ent0) if not rev else pos1_1 + len(ent1)
                    pos2_1 = len(sent0 + ent0 + sent1) if not rev else len(sent0)
                    pos2_2 = pos2_1 + len(ent1) if not rev else pos2_1 + len(ent0)
                    line['h']['pos'] = [pos1_1, pos1_2]
                    line['t']['pos'] = [pos2_1, pos2_2]
                    line['token'] = sent0 + ent0 + sent1 + ent1 + sent2
                    for j, t in enumerate(line['token']):
                        if t.startswith('##'):
                            line['token'][j] = t[2:]

                    ent_h_path, ent_t_path = dsp.parse(line['token'], line['h'], line['t'])
                    max_len = max(max_len, max(len(ent_h_path), len(ent_t_path)))
                    min_len = min(min_len, min(len(ent_h_path), len(ent_t_path)))
                    json.dump({'ent_h_path': ent_h_path, 'ent_t_path': ent_t_path}, f2, ensure_ascii=False)
                    f2.write('\n')
                    if (i + 1) % 100 == 0:
                        print(f'processed {i + 1} lines')
    print(max_len, min_len)

if __name__ == '__main__':
    # sent = "项目负责人应具有硕士以上学位"
    # nlp = StanfordCoreNLP('/home/liujian/NLP/corpus/tools/stanford-corenlp/stanford-corenlp-full-2018-10-05', lang='zh')
    # print(nlp.parse(sent))
    # pretrain_path = '/home/liujian/NLP/corpus/transformers/hfl-chinese-bert-wwm-ext'
    pretrain_path = '/home/liujian/NLP/corpus/transformers/google-bert-base-cased'
    semeval_path = os.path.abspath('../input/benchmark/relation/semeval')
    nyt10_path = os.path.abspath('../input/benchmark/relation/nyt10')
    wiki80_path = os.path.abspath('../input/benchmark/relation/wiki80')
    fewrel_path = os.path.abspath('../input/benchmark/relation/fewrel')
    # with bert tokenizer
    #print('*' * 20 + 'process semeval' + '*' * 20 + '\n')
    #construct_corpus_dsp(corpus_path=semeval_path, pretrain_path=pretrain_path, dsp_tool='stanza', language='en')
    #print('*' * 20 + 'process nyt10' + '*' * 20 + '\n')
    #construct_corpus_dsp(corpus_path=nyt10_path, pretrain_path=pretrain_path, dsp_tool='stanza', language='en')
    print('*' * 20 + 'process wiki80' + '*' * 20 + '\n')
    construct_corpus_dsp(corpus_path=wiki80_path, pretrain_path=pretrain_path, dsp_tool='stanza', language='en')
    print('*' * 20 + 'process fewrel' + '*' * 20 + '\n')
    construct_corpus_dsp(corpus_path=fewrel_path, pretrain_path=pretrain_path, dsp_tool='stanza', language='en')

                
