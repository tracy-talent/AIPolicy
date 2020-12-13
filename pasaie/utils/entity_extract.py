"""
 Author: liujian
 Date: 2020-11-03 10:21:15
 Last Modified by: liujian
 Last Modified time: 2020-11-03 10:21:15
"""

from collections import defaultdict

######################################
####### extract_kvpairs_by_tagscheme #######
######################################

# 严格按照BIO一致类型抽取实体
def extract_kvpairs_in_bio(bio_seq, word_seq):
    assert len(bio_seq) == len(word_seq)
    pairs = list()
    pre_bio = "O"
    v = ""
    spos = -1
    for i, bio in enumerate(bio_seq):
        word = word_seq[i]
        if bio == "O":
            if v != "": 
                pairs.append(((spos, i), pre_bio[2:], v))
            v = ""
        elif bio[0] == "B":
            if v != "": 
                pairs.append(((spos, i), pre_bio[2:], v))
            v = word[2:] if word.startswith('##') else word
            spos = i
        elif bio[0] == "I":
            if pre_bio[0] == "O" or pre_bio[2:] != bio[2:] or v == "":
                if v != "": 
                    pairs.append(((spos, i), pre_bio[2:], v))
                v = ""
            else:
                v += word[2:] if word.startswith('##') else word
        pre_bio = bio
    if v != "":
        pairs.append(((spos, len(bio_seq)), pre_bio[2:], v))
    return pairs

    
# 严格按照BIOE一致类型抽取实体
def extract_kvpairs_in_bioe(bioe_seq, word_seq):
    assert len(bioe_seq) == len(word_seq)
    pairs = list()
    pre_bioe = "O"
    v = ""
    spos = -1
    for i, bioe in enumerate(bioe_seq):
        word = word_seq[i]
        if bioe == "O":
            if v != "" and spos + 1 == i: 
                pairs.append(((spos, spos + 1), pre_bioe[2:], v))
            v = ""
        elif bioe[0] == "B":
            if v != "" and spos + 1 == i: 
                pairs.append(((spos, spos + 1), pre_bioe[2:], v))
            v = word[2:] if word.startswith('##') else word
            spos = i
        elif bioe[0] == "I":
            if pre_bioe[0] in "OE" or pre_bioe[2:] != bioe[2:] or v == "":
                if v != "" and spos + 1 == i: 
                    pairs.append(((spos, spos + 1), pre_bioe[2:], v))
                v = ""
            else:
                v += word[2:] if word.startswith('##') else word
        elif bioe[0] == 'E':
            if pre_bioe[0] in 'BI' and pre_bioe[2:] == bioe[2:] and v != "":
                v += word[2:] if word.startswith('##') else word
                pairs.append(((spos, i + 1), bioe[2:], v))
            v = ""
        pre_bioe = bioe
    if v != "" and spos + 1 == len(bioe_seq):
        pairs.append(((spos, spos + 1), pre_bioe[2:], v))
    return pairs


# 严格按照BIOES一致类型抽取实体
def extract_kvpairs_in_bioes(bioes_seq, word_seq):
    assert len(bioes_seq) == len(word_seq)
    pairs = list()
    pre_bioes = "O"
    v = ""
    spos = -1
    for i, bioes in enumerate(bioes_seq):
        word = word_seq[i]
        if bioes == "O":
            v = ""
        elif bioes[0] == "B":
            v = word[2:] if word.startswith('##') else word
            spos = i
        elif bioes[0] == "I":
            if pre_bioes[0] in "OES" or pre_bioes[2:] != bioes[2:] or v == "":
                v = ""
            else:
                v += word[2:] if word.startswith('##') else word
        elif bioes[0] == 'E':
            if pre_bioes[0] in 'BI' and pre_bioes[2:] == bioes[2:] and v != "":
                v += word[2:] if word.startswith('##') else word
                pairs.append(((spos, i + 1), bioes[2:], v))
            v = ""
        elif bioes[0] == 'S':
            v = word[2:] if word.startswith('##') else word
            pairs.append(((i, i + 1), bioes[2:], v))
            v = ""
        pre_bioes = bioes
    return pairs


# 严格按照BMOE一致类型抽取实体
def extract_kvpairs_in_bmoe(bioe_seq, word_seq):
    assert len(bioe_seq) == len(word_seq)
    pairs = list()
    pre_bioe = "O"
    v = ""
    spos = -1
    for i, bioe in enumerate(bioe_seq):
        word = word_seq[i]
        if bioe == "O":
            if v != "" and spos + 1 == i: 
                pairs.append(((spos, spos + 1), pre_bioe[2:], v))
            v = ""
        elif bioe[0] == "B":
            if v != "" and spos + 1 == i: 
                pairs.append(((spos, spos + 1), pre_bioe[2:], v))
            v = word[2:] if word.startswith('##') else word
            spos = i
        elif bioe[0] == "M":
            if pre_bioe[0] in "OE" or pre_bioe[2:] != bioe[2:] or v == "":
                if v != "" and spos + 1 == i: 
                    pairs.append(((spos, spos + 1), pre_bioe[2:], v))
                v = ""
            else:
                v += word[2:] if word.startswith('##') else word
        elif bioe[0] == 'E':
            if pre_bioe[0] in 'BM' and pre_bioe[2:] == bioe[2:] and v != "":
                v += word[2:] if word.startswith('##') else word
                pairs.append(((spos, i + 1), bioe[2:], v))
            v = ""
        pre_bioe = bioe
    if v != "" and spos + 1 == len(bioe_seq):
        pairs.append(((spos, spos + 1), pre_bioe[2:], v))
    return pairs


# 严格按照BMOES一致类型抽取实体
def extract_kvpairs_in_bmoes(bmoes_seq, word_seq):
    assert len(bmoes_seq) == len(word_seq)
    pairs = list()
    pre_bmoes = "O"
    v = ""
    spos = -1
    for i, bmoes in enumerate(bmoes_seq):
        word = word_seq[i]
        if bmoes == "O":
            v = ""
        elif bmoes[0] == "B":
            v = word[2:] if word.startswith('##') else word
            spos = i
        elif bmoes[0] == "M":
            if pre_bmoes[0] in "OES" or pre_bmoes[2:] != bmoes[2:] or v == "":
                v = ""
            else:
                v += word[2:] if word.startswith('##') else word
        elif bmoes[0] == 'E':
            if pre_bmoes[0] in 'BM' and pre_bmoes[2:] == bmoes[2:] and v != "":
                v += word[2:] if word.startswith('##') else word
                pairs.append(((spos, i + 1), bmoes[2:], v))
            v = ""
        elif bmoes[0] == 'S':
            v = word[2:] if word.startswith('##') else word
            pairs.append(((i, i + 1), bmoes[2:], v))
            v = ""
        pre_bmoes = bmoes
    return pairs


# 取实体最后一个词对应的分类结果，作为实体类型，应用于多任务中
def extract_kvpairs_in_bmoes_by_endtag(bioes_seq, word_seq, attr_seq):
    assert len(bioes_seq) == len(word_seq) == len(attr_seq)
    pairs = list()
    v = ""
    spos = -1
    for i in range(len(bioes_seq)):
        word = word_seq[i]
        bioes = bioes_seq[i]
        attr = attr_seq[i]
        if bioes == "O":
            v = ""
        elif bioes == "S":
            v = word[2:] if word.startswith('##') else word
            pairs.append(((i, i + 1), attr, v))
            v = ""
        elif bioes == "B":
            v = word[2:] if word.startswith('##') else word
            spos = i
        elif bioes == "M":
            if v != "": 
                v += word[2:] if word.startswith('##') else word
        elif bioes == "E":
            if v != "":
                v += word[2:] if word.startswith('##') else word
                pairs.append(((spos, i + 1), attr, v))
            v = ""
    return pairs


# 取实体最后一个词对应的分类结果，作为实体类型，应用于多任务中
def extract_kvpairs_in_bioes_by_endtag(bioes_seq, word_seq, attr_seq):
    assert len(bioes_seq) == len(word_seq) == len(attr_seq)
    pairs = list()
    v = ""
    spos = -1
    for i in range(len(bioes_seq)):
        word = word_seq[i]
        bioes = bioes_seq[i]
        attr = attr_seq[i]
        if bioes == "O":
            v = ""
        elif bioes == "S":
            v = word[2:] if word.startswith('##') else word
            pairs.append(((i, i + 1), attr, v))
            v = ""
        elif bioes == "B":
            v = word[2:] if word.startswith('##') else word
            spos = i
        elif bioes == "I":
            if v != "": 
                v += word[2:] if word.startswith('##') else word
        elif bioes == "E":
            if v != "":
                v += word[2:] if word.startswith('##') else word
                pairs.append(((spos, i + 1), attr, v))
            v = ""
    return pairs


# 选实体类别频率最高的类别作为实体类别
def extract_kvpairs_in_bmoes_by_vote(bioes_seq, word_seq, attr_seq):
    assert len(bioes_seq) == len(word_seq) == len(attr_seq)
    pairs = list()
    v = ""
    spos = -1
    for i in range(len(bioes_seq)):
        word = word_seq[i]
        bioes = bioes_seq[i]
        attr = attr_seq[i]
        if bioes == "O":
            v = ""
        elif bioes == "S":
            v = word[2:] if word.startswith('##') else word
            pairs.append(((i, i + 1), attr, v))
            v = ""
        elif bioes == "B":
            v = word[2:] if word.startswith('##') else word
            spos = i
        elif bioes == "M":
            if v != "": 
                v += word[2:] if word.startswith('##') else word
        elif bioes == "E":
            if v != "":
                v += word[2:] if word.startswith('##') else word
                vote = defaultdict(lambda: 0)
                for j in range(spos, i + 1):
                    vote[attr_seq[j]] += 1
                freq_attr = 'null'
                freq = 0
                for k in vote:
                    if vote[k] > freq:
                        freq = vote[k]
                        freq_attr = k
                pairs.append(((spos, i + 1), freq_attr, v))
            v = ""
    return pairs


# extract entities by start and end positions
def extract_kvpairs_by_start_end(start_seq, end_seq, word_seq, neg_tag):
    pairs = []
    for i, s_tag in enumerate(start_seq):
        if s_tag == neg_tag:
            continue
        for j, e_tag in enumerate(end_seq[i:]):
            if j > 0 and start_seq[j+i] != neg_tag:
                break
            if s_tag == e_tag:
                pairs.append(((i, j + i + 1), s_tag, ''.join([word[2:] if word.startswith('##') else word for word in word_seq[i:j+i+1]])))
                break
    return pairs