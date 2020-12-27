"""
 Author: liujian
 Date: 2020-12-27 20:57:50
 Last Modified by: liujian
 Last Modified time: 2020-12-27 20:57:50
"""

from .timedec import timeit

import os
import json

from ltp import LTP
from ddparser import DDParser
import stanza
from transformers import BertTokenizer


class Base_Parse(object):
    def __init__(self):
        pass

    def get_dependency_path(self, tokens: str, ent_h: dict, ent_t: dict, word: list, head: list, deprel: list):
        """get head entity to root dependency path, and get tail entity 
        to root dependency path

        Args:
            tokens (str): token list or sentence
            ent_h (dict): entity dict like {'pos': [hpos, tpos], 'entity': Time}
            ent_t (dict): entity dict like {'pos': [hpos, tpos], 'entity': Time}
            word (list): words
            head (list): head word, word[head[i] - 1] is the head word of word[i] 
            deprel (list): dependecy relation, deprel[i] is the dependency relation of word[head[i] - 1] to word[i]

        Returns:
            ent_h_path (list): DSP path of head entity to root
            ent_t_path (list): DSP path of tail entity to root
        """
        # construct map between word and token
        token2word = {}
        word2token = {}
        tpos, tlen = 0, 0
        wpos, wlen = 0, len(word[0])
        for i in range(len(tokens)):
            if tlen >= wlen:
                wpos += 1
                wlen += len(word[wpos])
                tpos = i
            tlen += len(tokens[i])
            token2word[i] = wpos
            word2token[wpos] = tpos

        # get entity word pos
        ent_h_word_pos_1 = token2word[ent_h['pos'][0]]
        ent_h_word_pos_2 = token2word[ent_h['pos'][1] - 1]
        ent_t_word_pos_1 = token2word[ent_t['pos'][0]]
        ent_t_word_pos_2 = token2word[ent_t['pos'][1] - 1]

        # get head entity dependency path to root
        ent_h_path = [ent_h_word_pos_2]
        while ent_h_path[-1] < len(head) and head[ent_h_path[-1]] != 0 and head[ent_h_path[-1]] - 1 not in ent_h_path:
            ent_h_path.append(head[ent_h_path[-1]] - 1)
        for i in range(len(ent_h_path)):
            ent_h_path[i] = word2token[ent_h_path[i]]
        # pos = [ent_h_word_pos_1]
        # while pos[-1] < ent_h_word_pos_2:
        #     if pos[-1] < len(head) and head[pos[-1]] - 1 <= ent_h_word_pos_2 and deprel[pos[-1]] == 'ATT' and head[pos[-1]] - 1 not in pos:
        #         pos.append(head[pos[-1]] - 1)
        #     else:
        #         break
        # if ent_h_word_pos_1 < pos[-1] and pos[-1] == ent_h_word_pos_2:
        #     ent_h_path[0] = word2token[ent_h_word_pos_1]

        # get tail entity dependency path to root
        ent_t_path = [ent_t_word_pos_2]
        while ent_t_path[-1] < len(head) and head[ent_t_path[-1]] != 0 and head[ent_t_path[-1]] - 1 not in ent_t_path:
            ent_t_path.append(head[ent_t_path[-1]] - 1)
        for i in range(len(ent_t_path)):
            ent_t_path[i] = word2token[ent_t_path[i]]
        # pos = [ent_t_word_pos_1]
        # while pos[-1] < ent_t_word_pos_2:
        #     if pos[-1] < len(head) and head[pos[-1]] - 1 <= ent_t_word_pos_2 and deprel[pos[-1]] == 'ATT' and head[pos[-1]] - 1 not in pos:
        #         pos.append(head[pos[-1]] - 1)
        #     else:
        #         break
        # if ent_t_word_pos_1 < pos[-1] and pos[-1] == ent_t_word_pos_2:
        #     ent_t_path[0] = word2token[ent_t_word_pos_1]
        
        return ent_h_path, ent_t_path


class DDP_Parse(Base_Parse):
    def __init__(self):
        super(DDP_Parse, self).__init__()
        self.ddp = DDParser()
    
    def parse(self, tokens, ent_h, ent_t):
        """get head entity to root dependency path, and get tail entity 
        to root dependency path

        Args:
            tokens (list or str): token list or sentence
            ent_h (dict): entity dict like {'pos': [hpos, tpos]}
            ent_t (dict): entity dict like {'pos': [hpos, tpos]}

        Returns:
            ent_h_path (list): DSP path of head entity to root
            ent_t_path (list): DSP path of tail entity to root
        """
        if isinstance(tokens, list):
            sent = ''.join(tokens)
        else:
            sent = tokens
        parse_dict = self.ddp.parse(sent)[0] # word, head, deprel
        # print(tokens[ent_h['pos'][0]:ent_h['pos'][1]], tokens[ent_t['pos'][0]:ent_t['pos'][1]])
        word = parse_dict['word']
        head = parse_dict['head']
        deprel = parse_dict['deprel']
        ent_h_path, ent_t_path = self.get_dependency_path(tokens, ent_h, ent_t, word, head, deprel)
        
        return ent_h_path, ent_t_path


class LTP_Parse(Base_Parse):
    def __init__(self):
        super(LTP_Parse, self).__init__()
        self.ltp = LTP()
    
    def parse(self, tokens, ent_h: dict, ent_t: dict):
        """get head entity to root dependency path, and get tail entity 
        to root dependency path

        Args:
            tokens (list or str): sentence
            ent_h (dict): entity dict like {'pos': [hpos, tpos]}
            ent_t (dict): entity dict like {'pos': [hpos, tpos]}

        Returns:
            ent_h_path (list): DSP path of head entity to root
            ent_t_path (list): DSP path of tail entity to root
        """
        if isinstance(tokens, list):
            sent = ''.join(tokens)
        else:
            sent = tokens
        seg, hidden = self.ltp.seg([sent])
        parse_tuple = self.ltp.dep(hidden)[0] # word, head, deprel
        parse_tuple = list(zip(*parse_tuple)) # [(tail token id), (head token id), (DSP relation)], id start from 1
        word = seg[0]
        head = parse_tuple[1]
        deprel = parse_tuple[2]
        ent_h_path, ent_t_path = self.get_dependency_path(tokens, ent_h, ent_t, word, head, deprel)
        
        return ent_h_path, ent_t_path


class Stanza_Parse(Base_Parse):
    def __init__(self):
        super(Stanza_Parse, self).__init__()
        self.nlp = stanza.Pipeline(lang='zh', processors='tokenize,pos,lemma,depparse')
    
    def parse(self, tokens, ent_h: dict, ent_t: dict):
        """get head entity to root dependency path, and get tail entity 
        to root dependency path

        Args:
            tokens (list or str): sentence
            ent_h (dict): entity dict like {'pos': [hpos, tpos]}
            ent_t (dict): entity dict like {'pos': [hpos, tpos]}

        Returns:
            ent_h_path (list): DSP path of head entity to root
            ent_t_path (list): DSP path of tail entity to root
        """
        if isinstance(tokens, list):
            sent = ''.join(tokens)
        else:
            sent = tokens
        doc = self.nlp(sent)
        triples = [(w.text, w.head, w.deprel) for sent in doc.sentences for w in sent.words]
        word, head, deprel = [list(seq) for seq in list(zip(*triples))]
        ent_h_path, ent_t_path = self.get_dependency_path(tokens, ent_h, ent_t, word, head, deprel)
        
        return ent_h_path, ent_t_path
        


@timeit
def construct_corpus_dsp(corpus_path, pretrain_path=None, dsp_tool='ddp'):
    """construct dependency syntax path of corpus, and save to file

    Args:
        corpus_path (str): corpus to be processed.
        pretrain_path (str, optional): bert pretrain path, can be bert, robert, albert and so on. Defaults to None.
        dsp_tool (str, optional): dsp tool be used to contruct dependency syntax path, choices:['ddp', 'ltp', 'stanza']. Defaults to 'ddp'.

    Raises:
        FileNotFoundError: raise when pretrain path is not found.
        NotImplementedError: raise when dsp_tool is not implemented.
    """
    if pretrain_path is not None and not os.path.exists(pretrain_path):
        raise FileNotFoundError(f'{pretrain_path} is not found, please give bert pretrain model path.')
    if pretrain_path is not None:
        tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        num_added_tokens = tokenizer.add_tokens(['“', '”', '—'])
    files = ['train', 'val', 'test']
    if dsp_tool == 'stanza':
        dsp = Stanza_Parse()
    elif dsp_tool == 'ltp':
        dsp = LTP_Parse()
    elif dsp_tool == 'ddp':
        dsp = DDP_Parse()
    else:
        raise NotImplementedError(f'{dsp_tool} DSP tool is not implemented, please use ltp, ddp or stanza!')
    corpus_name = corpus_path.split('/')[-1]
    max_len, min_len = 0, 1000000
    for fname in files:
        if not os.path.exists(os.path.join(corpus_path, f'{corpus_name}_{fname}.txt')):
            continue
        if pretrain_path is not None:
            dsp_file_name = f'{corpus_name}_{fname}_tail_bert_{dsp_tool}_dsp_path.txt'
        else:
            dsp_file_name = f'{corpus_name}_{fname}_tail_{dsp_tool}_dsp_path.txt'
        with open(os.path.join(corpus_path, f'{corpus_name}_{fname}.txt'), 'r', encoding='utf-8') as f1, open(os.path.join(corpus_path, dsp_file_name), 'w', encoding='utf-8') as f2:
            for i, line in enumerate(f1.readlines()):
                line = line.rstrip()
                if len(line) > 0:
                    line = eval(line)
                    if pretrain_path is not None:
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

        
        
        
        
