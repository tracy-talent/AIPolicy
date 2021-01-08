import io
import json
import os
import sys
from collections import deque

sys.path.append('..')
import pasaie
import re
from pasaie import pasaap, pasaner, pasare
from main import standard_ner_re_extraction


class FileMapper(object):

    def __init__(self, src_dir, dst_dir):
        self.name_mapper = self.file2id(src_dir, dst_dir)

    @staticmethod
    def file2id(src_dir, dst_dir):
        name_mapper = {}
        os.makedirs(dst_dir, exist_ok=True)
        for file in os.listdir(src_dir):
            if file.endswith(".txt") and file not in name_mapper:
                name_mapper[file] = str(len(name_mapper)) + '.txt'
                abs_file_path = os.path.join(src_dir, file)
                abs_write_path = os.path.join(dst_dir, name_mapper[file])

                # copy files
                try:
                    text = ''.join(open(abs_file_path, 'r', encoding='utf8').readlines())
                except:
                    try:
                        text = ''.join(open(abs_file_path, 'r', encoding='gbk').readlines())
                    except:
                        raise Exception(f"Cannot decode {file} in utf8 or gbk!")
                with open(abs_write_path, 'w', encoding='utf8') as fout:
                    fout.write(text)

        total_file_num = len([file for file in os.listdir(src_dir) if file.endswith('.txt')])

        file_mapper_fout = open(os.path.join(src_dir, 'file2id.json'), 'w', encoding='utf8')
        json.dump(name_mapper, file_mapper_fout, ensure_ascii=False, indent=1)
        print(f"Total files number: {total_file_num}; remain file number: {len(name_mapper)}; "
              f"overlap file number: {total_file_num - len(name_mapper)}")
        return name_mapper

    @staticmethod
    def generate_anns(data_dir, entity_model, relation_model, default_output_dir=None):
        if default_output_dir is None:
            default_output_dir = data_dir.replace('\\', '/') + '_targetTree'
        miss_sent_num, miss_entity_num, miss_relation_num = 0, 0, 0
        for file in os.listdir(data_dir):
            if file.endswith('.txt'):
                filepath = os.path.join(data_dir, file)
                ann_fout = open(os.path.join(data_dir, file.replace('.txt', '.ann')), 'w', encoding='utf8')
                # FIXME: The next line is in order to read file in Windows CRLF format
                text = io.open(filepath, 'rt', encoding='utf8', newline='').read()
                tree = pasaie.pasaap.tools.get_target_tree(filepath, output_dir=default_output_dir,
                                                           require_json=False, require_png=False)
                entity2tno = {}
                ann_entities, ann_relations = [], []

                queue = deque()
                queue.append(tree.root)
                while queue:
                    node = queue.popleft()
                    if node.is_root:
                        for name, child_node in node.children.items():
                            queue.append(child_node)
                    else:
                        sentence = node.sent
                        if sentence in text:
                            sent_index = text.index(sentence)
                            entities, relation_pairs, tokens = standard_ner_re_extraction(sentence,
                                                                                          entity_model,
                                                                                          relation_model)
                            entity_dict = {}
                            entity2entity = {}
                            for entity in entities:
                                pos_start, cate, entity_name = entity
                                if entity_name not in sentence:
                                    numbers = re.findall('\\d+', entity_name)
                                    if numbers:
                                        number_str = numbers[0]
                                        if entity_name.startswith(number_str):
                                            entity_name = entity_name.replace(number_str, f'{number_str} ')
                                        else:
                                            entity_name = entity_name.replace(number_str, f' {number_str} ')
                                spans = list(re.finditer(entity_name, sentence))
                                if spans:
                                    span_idx = entity_dict.get(entity_name, -1)
                                    bidx, eidx = spans[span_idx + 1].span()
                                    entity_dict[entity_name] = span_idx + 1
                                    format_entity = (cate, sent_index + bidx, sent_index + eidx,
                                                     text[sent_index + bidx: sent_index + eidx])
                                    ann_entities.append(format_entity)
                                    entity2entity[tuple(entity)] = format_entity
                                else:
                                    miss_entity_num += 1
                                    print(f"Cannot find entity {entity} in {sentence}")
                            for relation in relation_pairs:
                                head, tail, rtype = relation
                                ann_relations.append((entity2entity[head], entity2entity[tail], rtype))
                        else:
                            miss_sent_num += 1

                # write ann
                for ith, _entity in enumerate(sorted(ann_entities, key=lambda x: (x[1], x[2])), 1):
                    _cate, _start, _end, _entity_name = _entity
                    entity2tno[_entity] = ith
                    ann_fout.write('T{}\t{} {} {}\t{}\n'.format(str(ith), _cate, _start, _end, _entity_name))
                for ith, _relation in enumerate(sorted(ann_relations, key=lambda x: x[0][0]), 1):
                    head, tail, rtype = _relation
                    if head not in entity2tno or tail not in entity2tno:
                        miss_relation_num += 1
                        continue
                    head_tno, tail_tno = entity2tno[head], entity2tno[tail]
                    ann_fout.write('R{}\t{} Arg1:T{} Arg2:T{}\t\n'.format(ith, rtype, head_tno, tail_tno))

        print("miss sentence: {}; miss entity: {}; miss relation: {}".format(miss_sent_num,
                                                                             miss_entity_num, miss_relation_num))


def idx2rawidx(tokens, raw_sentence):
    idx2idx = {}
    idx1, idx2 = 0, 0
    # 还要考虑token可能是像日期“2015”
    while idx1 < len(tokens) or idx2 < len(raw_sentence):
        if tokens[idx1].startswith("##") or tokens[idx1].strip() == '':
            idx2idx[idx1] = idx2
            idx1 += 1
        elif raw_sentence[idx2].strip() == '':
            idx2 += 1
        elif tokens[idx1] in raw_sentence[idx2] and idx1 + 1 < len(tokens) and tokens[idx1 + 1].startswith("##"):
            idx2idx[idx1] = idx2
            idx1 += 1
        elif tokens[idx1] == raw_sentence[idx2]:
            idx2idx[idx1] = idx2
            idx1 += 1
            idx2 += 1
        else:
            raise Exception(tokens[idx1], raw_sentence[idx2])
    return idx2idx


if __name__ == '__main__':
    # file_mapper = FileMapper(src_dir=r'C:\Users\90584\Desktop\政策实体与关系抽取\语料\南京-大全txt',
    #                          dst_dir=r'C:\Users\90584\Desktop\政策实体与关系抽取\语料\policies_ext')
    # text = ''.join(fin.readlines())
    # print(len(text))
    my_entity_model = pasaie.pasaner.get_model('policy_bmoes/bert_lstm_crf0')
    my_relation_model = pasaie.pasare.get_model('test-policy/bert_entity_dice_alpha0.6_fgm0')
    FileMapper.generate_anns(data_dir=r'./selection', entity_model=my_entity_model, relation_model=my_relation_model)
