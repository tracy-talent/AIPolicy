import os
import re
from collections import defaultdict

import numpy as np
from prettytable import PrettyTable
from colorama import init, Fore, Back, Style
init(autoreset=False)

# 以下字符串出现顺序需要与log文件中一致，否则会因匹配优先级出错
target_score_strings = ['Span Accuracy', 'Span Micro precision', 'Span Micro recall', 'Span Micro F1',
                        'Micro precision', 'Micro recall', 'Micro F1']
dpr_regex = 'dpr(\d+\.\d+)'
wz_regex = '\(w(\d+),'
test_set_regex = 'Test set results'
pinyin_vec_regex = '--pinyin2vec_file\s.*\\(.*)\.vec'


def extract_score_from_logs(log_dir, save_path=None):
    dataset_dirs = [_dir for _dir in os.listdir(log_dir)
                    if os.path.isdir(os.path.join(log_dir, _dir))]

    dataset_scores = {}
    for dataset in dataset_dirs:
        dataset_log_path = os.path.join(log_dir, dataset)
        param_dict = {}
        for subdir in os.listdir(dataset_log_path):
            subdir_path = os.path.join(dataset_log_path, subdir)
            target_scores = defaultdict(list)
            pinyin_vec = None
            for logfile in os.listdir(subdir_path):
                if logfile.endswith('.log'):
                    score_dict, dpr, wz, pinyin_vec = extract_target_scores(os.path.join(subdir_path, logfile))
                    if all(score_str in score_dict for score_str in target_score_strings):
                        target_scores['dpr'].append(float(dpr))
                        target_scores['wz'].append(float(wz))
                        for score_name, score in score_dict.items():
                            target_scores[score_name].append(float(score))

            # 对同一参数目录下的所有log文件取均值
            tmp_dict = {}
            for score_name, score_list in target_scores.items():
                if score_name in target_score_strings:
                    factor = 100
                else:
                    factor = 1
                tmp_dict[score_name] = round(np.mean(score_list) * factor, ndigits=2)
            if len(tmp_dict) > 0:
                tmp_dict['pinyin_vec'] = pinyin_vec
                param_dict[subdir] = tmp_dict
        dataset_scores[dataset] = param_dict

    table_list = display_results(dataset_scores)
    if save_path:
        with open(save_path, 'w', encoding='utf8') as fout:
            fout.write('\n'.join(table_list))
    else:
        print('\n'.join(table_list))


def extract_target_scores(log_path):
    score_dict = {}
    dpr, wz = None, None
    is_finish_test = False
    pinyin_vec = None
    with open(log_path, 'r', encoding='utf8') as fin:
        for ith, line in enumerate(fin.readlines()):
            line = line.strip()
            if ith == 0:
                vec_file = re.findall('--pinyin2vec_file\s.*/(.*)\.vec', line)
                if vec_file:
                    pinyin_vec = vec_file[0]
            if re.match(f'.*{test_set_regex}', line):
                is_finish_test = True

            if is_finish_test:
                for match_str in target_score_strings:
                    if re.match(f'.*{match_str}', line):
                        score_dict[match_str] = re.findall(f'{match_str}:\s(\d+\.\d+)', line)[0]

                if re.match(f'.*{dpr_regex}', line):
                    dpr = re.findall(dpr_regex, line)[0]
                if re.match(f'.*{wz_regex}', line):
                    wz = re.findall(wz_regex, line)[0]

    return score_dict, dpr, wz, pinyin_vec


# 从目录中获取我们需要的信息
def parse_params_dir(params_dir_str, pinyin_vec):
    group_name = params_dir_str.split('_', maxsplit=1)[0]
    lexicon_dict = 'ctb' if 'ctb' in params_dir_str else 'sgns'
    return '_'.join([group_name, lexicon_dict, pinyin_vec])


def resolve_data(param_dict):
    ret_list = []
    color = Colored()
    precision_list, recall_list, f1_list = [], [], []
    for _, score_dict in param_dict.items():
        precision_list.append(score_dict['Micro precision'])
        recall_list.append(score_dict['Micro recall'])
        f1_list.append(score_dict['Micro F1'])
    if len(precision_list) == 0 or len(recall_list) == 0 or len(f1_list) == 0:
        return []
    precision_max = max(precision_list)
    recall_max = max(recall_list)
    f1_max = max(f1_list)
    
    for param_name, score_dict in param_dict.items():
        if score_dict['pinyin_vec'] is None:
            continue
        simplify_param = parse_params_dir(param_name, score_dict['pinyin_vec'])
        simplify_param += color.red(f"(dpr={score_dict['dpr']},wz={int(score_dict['wz'])})")
        span_micro_p = "%.2f" % score_dict['Span Micro precision']
        span_micro_r = "%.2f" % score_dict['Span Micro recall']
        span_micro_f1 = "%.2f" % score_dict['Span Micro F1']
        micro_p = "%.2f" % score_dict['Micro precision']
        micro_r = "%.2f" % score_dict['Micro recall']
        micro_f1 = "%.2f" % score_dict['Micro F1']
        micro_p = color.red(micro_p) if float(micro_p) == precision_max else micro_p
        micro_r = color.red(micro_r) if float(micro_r) == recall_max else micro_r
        micro_f1 = color.red(micro_f1) if float(micro_f1) == f1_max else micro_f1
        ret_list.append([simplify_param,
                         span_micro_p + '/' + micro_p,
                         span_micro_r + '/' + micro_r,
                         span_micro_f1 + '/' + micro_f1
                         ])
    return ret_list


def display_results(res_dict):
    table_list = []
    for dataset, param_dict in res_dict.items():

        table = PrettyTable(['params', 'Precision', 'Recall', 'F1'])
        rows = resolve_data(param_dict)
        for row in rows:
            table.add_row(row)
        table_string = '='*100 + '\n' + dataset + '\n' + table.get_string()
        table_list.append(table_string)
    return table_list


class Colored(object):
    #  前景色:红色  背景色:默认
    def red(self, s):
        return Fore.LIGHTRED_EX + s + Fore.RESET

    #  前景色:绿色  背景色:默认
    def green(self, s):
        return Fore.LIGHTGREEN_EX + s + Fore.RESET

    def yellow(self, s):
        return Fore.LIGHTYELLOW_EX + s + Fore.RESET

    def white(self, s):
        return Fore.LIGHTWHITE_EX + s + Fore.RESET

    def blue(self, s):
        return Fore.LIGHTBLUE_EX + s + Fore.RESET


if __name__ == '__main__':
    extract_score_from_logs(r'C:\NLP-Github\AIPolicy\output_tmp\entity\logs',
                            save_path=None)
