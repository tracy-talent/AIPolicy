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
    k = ""
    spos = -1
    for i, bio in enumerate(bio_seq):
        if bio == "O":
            if v != "" and k[0] == "B": 
                pairs.append((spos, pre_bio[2:], v))
            v = ""
            k = ""
        elif bio[0] == "B":
            if v != "" and k[0] == "B": 
                pairs.append((spos, pre_bio[2:], v))
            v = word_seq[i]
            k = "B"
            spos = i
        elif bio[0] == "I":
            if pre_bio[0] == "O" or pre_bio[2:] != bio[2:]:
                if v != "" and k[0] == "B": 
                    pairs.append((spos, pre_bio[2:], v))
                v = ""
                k = ""
            else:
                v += word_seq[i]
                k += "I"
        pre_bio = bio
    if v != "" and k[0] == "B":
        pairs.append((spos, pre_bio[2:], v))
    return pairs

    
# 严格按照BIOE一致类型抽取实体
def extract_kvpairs_in_bioe(bioe_seq, word_seq):
    assert len(bioe_seq) == len(word_seq)
    pairs = list()
    pre_bioe = "O"
    v = ""
    k = ""
    spos = -1
    for i, bioe in enumerate(bioe_seq):
        if bioe == "O":
            if v != "" and len(k) == 1 and k[0] == "B": 
                pairs.append((spos, pre_bioe[2:], v))
            v = ""
            k = ""
        elif bioe[0] == "B":
            if v != "" and len(k) == 1 and k[0] == "B": 
                pairs.append((spos, pre_bioe[2:], v))
            v = word_seq[i]
            k = "B"
            spos = i
        elif bioe[0] == "I":
            if pre_bioe[0] in "OE" or pre_bioe[2:] != bioe[2:]:
                if v != "" and len(k) == 1 and k[0] == "B": 
                    pairs.append((spos, pre_bioe[2:], v))
                v = ""
                k = ""
            else:
                v += word_seq[i]
                k += "I"
        elif bioe[0] == 'E':
            if pre_bioe[0] in 'BI' and pre_bioe[2:] == bioe[2:] and v != "" and k[0] == "B":
                v += word_seq[i]
                pairs.append((spos, bioe[2:], v))
            v = ""
            k = ""
        pre_bioe = bioe
    if v != "" and len(k) == 1 and k[0] == "B":
        pairs.append((spos, pre_bioe[2:], v))
    return pairs


# 严格按照BIOES一致类型抽取实体
def extract_kvpairs_in_bioes(bioes_seq, word_seq):
    assert len(bioes_seq) == len(word_seq)
    pairs = list()
    pre_bioes = "O"
    v = ""
    k = ""
    spos = -1
    for i, bioes in enumerate(bioes_seq):
        if bioes == "O":
            v = ""
            k = ""
        elif bioes[0] == "B":
            v = word_seq[i]
            k = "B"
            spos = i
        elif bioes[0] == "I":
            if pre_bioes[0] in "OES" or pre_bioes[2:] != bioes[2:]:
                v = ""
                k = ""
            else:
                v += word_seq[i]
                k += "I"
        elif bioes[0] == 'E':
            if pre_bioes[0] in 'BI' and pre_bioes[2:] == bioes[2:] and v != "" and k[0] == "B":
                v += word_seq[i]
                pairs.append((spos, bioes[2:], v))
            v = ""
            k = ""
        elif bioes[0] == 'S':
            pairs.append((i, bioes[2:], word_seq[i]))
            v = ""
            k = ""
        pre_bioes = bioes
    return pairs


# 严格按照BMOE一致类型抽取实体
def extract_kvpairs_in_bmoe(bioe_seq, word_seq):
    assert len(bioe_seq) == len(word_seq)
    pairs = list()
    pre_bioe = "O"
    v = ""
    k = ""
    spos = -1
    for i, bioe in enumerate(bioe_seq):
        if bioe == "O":
            if v != "" and len(k) == 1 and k[0] == "B": 
                pairs.append((spos, pre_bioe[2:], v))
            v = ""
            k = ""
        elif bioe[0] == "B":
            if v != "" and len(k) == 1 and k[0] == "B": 
                pairs.append((spos, pre_bioe[2:], v))
            v = word_seq[i]
            k = "B"
            spos = i
        elif bioe[0] == "M":
            if pre_bioe[0] in "OE" or pre_bioe[2:] != bioe[2:]:
                if v != "" and len(k) == 1 and k[0] == "B": 
                    pairs.append((spos, pre_bioe[2:], v))
                v = ""
                k = ""
            else:
                v += word_seq[i]
                k += "M"
        elif bioe[0] == 'E':
            if pre_bioe[0] in 'BM' and pre_bioe[2:] == bioe[2:] and v != "" and k[0] == "B":
                v += word_seq[i]
                pairs.append((spos, bioe[2:], v))
            v = ""
            k = ""
        pre_bioe = bioe
    if v != "" and len(k) == 1 and k[0] == "B":
        pairs.append((spos, pre_bioe[2:], v))
    return pairs


# 严格按照BMOES一致类型抽取实体
def extract_kvpairs_in_bmoes(bmoes_seq, word_seq):
    assert len(bmoes_seq) == len(word_seq)
    pairs = list()
    pre_bmoes = "O"
    v = ""
    k = ""
    spos = -1
    for i, bmoes in enumerate(bmoes_seq):
        if bmoes == "O":
            v = ""
            k = ""
        elif bmoes[0] == "B":
            v = word_seq[i]
            k = "B"
            spos = i
        elif bmoes[0] == "M":
            if pre_bmoes[0] in "OES" or pre_bmoes[2:] != bmoes[2:]:
                v = ""
                k = ""
            else:
                v += word_seq[i]
                k += "M"
        elif bmoes[0] == 'E':
            if pre_bmoes[0] in 'BM' and pre_bmoes[2:] == bmoes[2:] and v != "" and k[0] == "B":
                v += word_seq[i]
                pairs.append((spos, bmoes[2:], v))
            v = ""
            k = ""
        elif bmoes[0] == 'S':
            pairs.append((i, bmoes[2:], v))
            v = ""
            k = ""
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
            v = word
            pairs.append((i, attr, v))
            v = ""
        elif bioes == "B":
            v = word
            spos = i
        elif bioes == "M":
            if v != "": 
                v += word
        elif bioes == "E":
            if v != "":
                v += word
                pairs.append((spos, attr, v))
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
            v = word
            pairs.append((i, attr, v))
            v = ""
        elif bioes == "B":
            v = word
            spos = i
        elif bioes == "I":
            if v != "": 
                v += word
        elif bioes == "E":
            if v != "":
                v += word
                pairs.append((spos, attr, v))
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
            v = word
            pairs.append((i, attr, v))
            v = ""
        elif bioes == "B":
            v = word
            spos = i
        elif bioes == "M":
            if v != "": 
                v += word
        elif bioes == "E":
            if v != "":
                v += word
                vote = defaultdict(lambda: 0)
                for j in range(spos, i + 1):
                    vote[attr_seq[j]] += 1
                freq_attr = 'null'
                freq = 0
                for k in vote:
                    if vote[k] > freq:
                        freq = vote[k]
                        freq_attr = k
                pairs.append((spos, freq_attr, v))
            v = ""
    return pairs


# extract entities by start and end positions
def extract_kvpairs_by_start_end(start_seq, end_seq, word_seq, neg_tag):
    pairs = []
    for i, s_tag in enumerate(start_seq):
        if s_tag == neg_tag:
            continue
        for j, e_tag in enumerate(end_seq[i:]):
            if s_tag == e_tag:
                pairs.append((i, s_tag, ''.join(word_seq[i:j+1])))
                break
    return pairs