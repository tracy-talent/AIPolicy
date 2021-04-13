import os
from ast import literal_eval
from collections import defaultdict


def search_and_save_error_case(case_study_filepath, dataset):
    train_path = f'../input/benchmark/entity/{dataset}/train.char.bmoes'
    entity_dict = get_nertext_entity_dict(train_path)
    error_num, unseen_num = 0, 0
    total_unseen, correct_unseen = 0, 0

    output_path = case_study_filepath.replace('.txt', '_error.txt')
    with open(case_study_filepath, 'r', encoding='utf8') as fin:
        error_examples = []
        content = fin.readlines()
        for ith in range(len(content) // 4):
            text, gold, pred = content[ith*4+0].strip(), literal_eval(content[ith*4+1].strip()), literal_eval(content[ith*4+2].strip())
            unseen_entities = [ele for ele in gold if ele[2] not in entity_dict[ele[1]]]
            total_unseen += len(unseen_entities)
            correct_unseen += len([ele for ele in unseen_entities if ele in pred])
            if len(gold) != len(pred) or any(ele not in pred for ele in gold):
                diff_list = list(set(gold).difference(set(pred)))
                unseen_ent_list = []
                for term in diff_list:
                    pos, tag, ent = term
                    if ent not in entity_dict[tag]:
                        unseen_ent_list.append(term)

                error_num += len(diff_list)
                unseen_num += len(unseen_ent_list)
                error_examples.append(f'{text}\ngold: {gold}\npred: {pred}\ndiff: {diff_list}\nunseen: {unseen_ent_list}')

        with open(output_path, 'w', encoding='utf8') as fout:
            fout.write('\n\n'.join(error_examples))

        filepath = '/'.join(case_study_filepath.replace('\\', '/').split('/')[-3:])
        print(f"filepath:\t{filepath}\n"
              f"unseen_correct rate: {round(correct_unseen/total_unseen * 100, 3)}%;"
              f" unseen/error rate: {round(unseen_num / error_num * 100, 3)}%")


# BMOES tag
def nertext2flagtext(nertext_path):
    texts = []
    ins = []
    for line in open(nertext_path, 'r', encoding='utf8'):
        line = line.strip()
        if line:
            ins.append(line.split()[0])
        else:
            texts.append(''.join(ins))
            ins = []
    return texts

# BMOES tag
def get_nertext_entity_dict(nertext_path):
    entity_dict = defaultdict(set)
    entity_str = ''
    entity_type = None
    for line in open(nertext_path, 'r', encoding='utf8'):
        line = line.strip()
        if line:
            ch, tag = line.split()
            if tag.startswith('B'):
                if entity_str:
                    entity_dict[entity_type].add(entity_str)
                entity_type = tag[2:]
                entity_str = ch
            elif tag.startswith('O'):
                if entity_str and entity_type:
                    entity_dict[entity_type].add(entity_str)

                entity_type = ''
                entity_str = ''
            else:
                entity_str += ch
    if entity_str:
        entity_dict[entity_type].add(entity_str)
    return entity_dict


dataset_dir = r'C:\Users\90584\Desktop\AIPolicy实验\AIPolicy\output\entity\logs\msra_bmoes'
for subdir in os.listdir(dataset_dir):
    dataset = dataset_dir.replace('\\', '/').split('/')[-1].split('_')[0]
    for file in os.listdir(os.path.join(dataset_dir, subdir)):
        if 'case_study' in file and 'error' not in file:
            filepath = os.path.join(dataset_dir, subdir, file)
            search_and_save_error_case(filepath, dataset)

# search_and_save_error_case(
#     case_study_filepath=filepath,
#     dataset='msra'
# )