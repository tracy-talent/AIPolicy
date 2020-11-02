"""
 Author: liujian
 Date: 2020-10-27 23:07:40
 Last Modified by: liujian
 Last Modified time: 2020-10-27 23:07:40
"""

from contextlib import ExitStack


def convert_bio_to_bmoes(in_file, out_file):
    """convert bio tag scheme to bmoes tag scheme

    Args:
        in_file (str): bio taged sequence file
        out_file (str): bmoes taged sequence file
    """
    with open(in_file, 'r', encoding='utf-8') as f1, open(out_file, 'w', encoding='utf-8') as f2:
        sent = []
        for line in f1:
            line = line.strip().split()
            if len(line) == 0 and len(sent) > 0:
                for i, item in enumerate(sent):
                    if item[1][0] == 'I':
                        if i + 1 >= len(sent) or sent[i + 1][1][0] != 'I':
                            item[1] = 'E' + item[1][1:]
                        else:
                            item[1] = 'M' + item[1][1:]
                    elif item[1][0] == 'B':
                        if i + 1 >= len(sent) or sent[i + 1][1][0] != 'I':
                            item[1] = 'S' + item[1][1:]
                for item in sent:
                    f2.write(' '.join(item) + '\n')
                f2.write('\n')
                sent = []
            else:
                sent.append(line)


def convert_bio_to_bmoe(in_file, out_file):
    """convert bio tag scheme to bmoe tag scheme

    Args:
        in_file (str): bio taged sequence file
        out_file (str): bmoe taged sequence file
    """
    with open(in_file, 'r', encoding='utf-8') as f1, open(out_file, 'w', encoding='utf-8') as f2:
        sent = []
        for line in f1:
            line = line.strip().split()
            if len(line) == 0 and len(sent) > 0:
                for i, item in enumerate(sent):
                    if item[1][0] == 'I':
                        if i + 1 >= len(sent) or sent[i + 1][1][0] != 'I':
                            item[1] = 'E' + item[1][1:]
                        else:
                            item[1] = 'M' + item[1][1:]
                for item in sent:
                    f2.write(' '.join(item) + '\n')
                f2.write('\n')
                sent = []
            else:
                sent.append(line)


def convert_bio_to_bioes(in_file, out_file):
    """convert bio tag scheme to bioes tag scheme

    Args:
        in_file (str): bio taged sequence file
        out_file (str): bioes taged sequence file
    """
    with open(in_file, 'r', encoding='utf-8') as f1, open(out_file, 'w', encoding='utf-8') as f2:
        sent = []
        for line in f1:
            line = line.strip().split()
            if len(line) == 0 and len(sent) > 0:
                for i, item in enumerate(sent):
                    if item[1][0] == 'I':
                        if i + 1 >= len(sent) or sent[i + 1][1][0] != 'I':
                            item[1] = 'E' + item[1][1:]
                    elif item[1][0] == 'B':
                        if i + 1 >= len(sent) or sent[i + 1][1][0] != 'I':
                            item[1] = 'S' + item[1][1:]
                for item in sent:
                    f2.write(' '.join(item) + '\n')
                f2.write('\n')
                sent = []
            else:
                sent.append(line)


def convert_bio_to_bioe(in_file, out_file):
    """convert bio tag scheme to bioe tag scheme

    Args:
        in_file (str): bio taged sequence file
        out_file (str): bioe taged sequence file
    """
    with open(in_file, 'r', encoding='utf-8') as f1, open(out_file, 'w', encoding='utf-8') as f2:
        sent = []
        for line in f1:
            line = line.strip().split()
            if len(line) == 0 and len(sent) > 0:
                for i, item in enumerate(sent):
                    if item[1][0] == 'I':
                        if i + 1 >= len(sent) or sent[i + 1][1][0] != 'I':
                            item[1] = 'E' + item[1][1:]
                for item in sent:
                    f2.write(' '.join(item) + '\n')
                f2.write('\n')
                sent = []
            else:
                sent.append(line)


def convert_bmoes_to_bio(in_file, out_file):
    """convert bmoes tag scheme to bio tag scheme

    Args:
        in_file (str): bmoes taged sequence file
        out_file (str): bio taged sequence file
    """
    with open(in_file, 'r', encoding='utf-8') as f1, open(out_file, 'w', encoding='utf-8') as f2:
        sent = []
        for line in f1:
            line = line.strip().split()
            if len(line) == 0 and len(sent) > 0:
                for i, item in enumerate(sent):
                    if item[1][0] == 'M':
                        item[1] = 'I' + item[1][1:]
                    elif item[1][0] == 'E':
                        item[1] = 'I' + item[1][1:]
                    elif item[1][0] == 'S':
                        item[1] = 'B' + item[1][1:]   
                for item in sent:
                    f2.write(' '.join(item) + '\n')
                f2.write('\n')
                sent = []
            else:
                sent.append(line)


def construct_tag2id_bmoes(corpus_file, out_file):
    """construct bmoes tag2id file

    Args:
        corpus_file (str): corpus file path
        out_file (str): tag2id.bmoes
    """
    tagset = set()
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for lid, line in enumerate(f):
            line = line.strip()
            seq_bio = []
            f = False
            f1 = False
            for i in range(len(line)):
                if line[i] == '(' and not f:
                    s = i
                    f = True
                elif line[i] == ')' and f1:
                    seq_bio.append(list(eval(line[s:i+1])))
                    f = False
                    f1 = False
                elif line[i] == ',' and line[i:i+2] == ', ':
                    f1 = True
            for i, item in enumerate(seq_bio):
                if item[1][0] == 'I':
                    if i + 1 >= len(seq_bio) or seq_bio[i + 1][1][0] != 'I':
                        item[1] = 'E' + item[1][1:]
                    else:
                        item[1] = 'M' + item[1][1:]
                elif item[1][0] == 'B':
                    if i + 1 >= len(seq_bio) or seq_bio[i + 1][1][0] != 'I':
                        item[1] = 'S' + item[1][1:]
            for _, tag in seq_bio:
                tagset.add(tag)
    with open(out_file, 'w', encoding='utf-8') as f:
        tagset.remove('O')
        attrset = set()
        for tag in tagset:
            attrset.add(tag[2:])
        for attr in attrset:
            if ('B-' + attr) in tagset:
                f.write('B-' + attr + '\n')
            if ('M-' + attr) in tagset:
                f.write('M-' + attr + '\n')
            if ('E-' + attr) in tagset:
                f.write('E-' + attr + '\n')
            if ('S-' + attr) in tagset:
                f.write('S-' + attr + '\n')
        f.write('O')


def construct_span2id_and_attr2id(tag2id_file, span2id_file, attr2id_file):
    """construct span2id and attr2id file from tag2id file

    Args:
        tag2id_file (str): file path of tag2id
        span2id_file (str): file path of span2id
        attr2id_file (str): file path of attr2id
    """
    with ExitStack() as stack:
        file_name_list = [span2id_file, attr2id_file]
        span_attr_file = [stack.enter_context(open(fname, 'w', encoding='utf-8')) for fname in file_name_list]
        spanset = set()
        attrset = set()
        with open(tag2id_file, 'r', encoding='utf-8') as f:
            for tag in f:
                tag = tag.strip()
                if len(tag) == 1:
                    spanset.add(tag)
                else:
                    spanset.add(tag[0])
                    attrset.add(tag[2:])
        spanset.remove('O')
        for span in spanset:
            span_attr_file[0].write(span + '\n')
        span_attr_file[0].write('O')
        for attr in attrset:
            span_attr_file[1].write(attr + '\n')
        span_attr_file[1].write('null')