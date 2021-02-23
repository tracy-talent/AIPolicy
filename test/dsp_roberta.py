import configparser

import os
import json

import sys
sys.path.append('..')
from pasaie.utils.dependency_parse import LTP_Parse, DDP_Parse, Stanza_Parse
from pasaie.utils.dependency_parse import construct_corpus_dsp_with_bert, construct_corpus_dsp_with_xlnet
from pasaie.utils.timedec import timeit
from transformers import BertTokenizer

if __name__ == '__main__':
    # sent = "项目负责人应具有硕士以上学位"
    # nlp = StanfordCoreNLP('/home/liujian/NLP/corpus/tools/stanford-corenlp/stanford-corenlp-full-2018-10-05', lang='zh')
    # print(nlp.parse(sent))
    # pretrain_path = '/home/liujian/NLP/corpus/transformers/hfl-chinese-roberta-base-wwm-ext' # zh
    # pretrain_path = '/home/liujian/NLP/corpus/transformers/hfl-chinese-roberta-large-wwm-ext' # zh
    # pretrain_path = '/home/liujian/NLP/corpus/transformers/roberta-base-cased' # en
    pretrain_path = '/home/liujian/NLP/corpus/transformers/roberta-large-cased' # en
    semeval_path = os.path.abspath('../input/benchmark/relation/semeval')
    #nyt10_path = os.path.abspath('../input/benchmark/relation/nyt10')
    #wiki80_path = os.path.abspath('../input/benchmark/relation/wiki80')
    #fewrel_path = os.path.abspath('../input/benchmark/relation/fewrel')
    kbp37_path = os.path.abspath('../input/benchmark/relation/kbp37')
    # nyt24_path = os.path.abspath('../input/benchmark/relation/nyt24')
    # nyt29_path = os.path.abspath('../input/benchmark/relation/nyt29')
    # with bert tokenizer
    print('*' * 20 + 'process semeval' + '*' * 20 + '\n')
    construct_corpus_dsp_with_bert(corpus_path=semeval_path, pretrain_path=pretrain_path, dsp_tool='stanza', language='en')
    #print('*' * 20 + 'process nyt10' + '*' * 20 + '\n')
    #construct_corpus_dsp(corpus_path=nyt10_path, pretrain_path=pretrain_path, dsp_tool='stanza', language='en')
    #print('*' * 20 + 'process wiki80' + '*' * 20 + '\n')
    #construct_corpus_dsp(corpus_path=wiki80_path, pretrain_path=pretrain_path, dsp_tool='stanza', language='en')
    #print('*' * 20 + 'process fewrel' + '*' * 20 + '\n')
    #construct_corpus_dsp(corpus_path=fewrel_path, pretrain_path=pretrain_path, dsp_tool='stanza', language='en')
    print('*' * 20 + 'process kbp37' + '*' * 20 + '\n')
    construct_corpus_dsp_with_bert(corpus_path=kbp37_path, pretrain_path=pretrain_path, dsp_tool='stanza', language='en')
    # print('*' * 20 + 'process nyt24' + '*' * 20 + '\n')
    # construct_corpus_dsp(corpus_path=nyt24_path, pretrain_path=pretrain_path, dsp_tool='stanza', language='en')
    # print('*' * 20 + 'process nyt29' + '*' * 20 + '\n')
    # construct_corpus_dsp(corpus_path=nyt29_path, pretrain_path=pretrain_path, dsp_tool='stanza', language='en')

