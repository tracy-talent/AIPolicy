import os
import re
import xlwt
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


def extract_score_from_logs(log_dir, save_path=None, target_dataset="all"):
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
                    if dpr is None or wz is None:
                        continue
                    if pinyin_vec is None and 'char' in subdir:
                        pinyin_vec = 'char'
                    if all(score_str in score_dict for score_str in target_score_strings):
                        target_scores['dpr'].append(float(dpr))
                        target_scores['wz'].append(float(wz))
                        for score_name, score in score_dict.items():
                            target_scores[score_name].append(float(score))

            # 对同一参数目录下的所有log文件取均值，factor只是为了表示为%
            tmp_dict = {}
            for score_name, score_list in target_scores.items():
                if score_name in target_score_strings:
                    score_list = [s * 100 for s in score_list]
                tmp_dict[score_name] = round(np.mean(score_list), ndigits=2)

            if len(tmp_dict) > 0:
                if pinyin_vec is None:
                    pinyin_vec = 'char'
                    print(subdir_path)
                tmp_dict['pinyin_vec'] = pinyin_vec
                param_dict[subdir] = tmp_dict
        dataset_scores[dataset] = param_dict

    table_list = display_results(dataset_scores, target_dataset)
    if save_path:
        save_as_excel(dataset_scores, save_path=save_path)
        print('\n'.join(table_list))
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
                        try:
                            score_dict[match_str] = re.findall(f'{match_str}:\s(\d+\.\d+)', line)[0]

                        except:
                            score_dict[match_str] = re.findall(f'{match_str}:\s(\d+\.?\d*)', line)[0]   # 匹配无小数点的数字

                if re.match(f'.*{dpr_regex}', line):
                    dpr = re.findall(dpr_regex, line)[0]
                if re.match(f'.*{wz_regex}', line):
                    wz = re.findall(wz_regex, line)[0]

    return score_dict, dpr, wz, pinyin_vec


# 从目录中获取我们需要的信息
def parse_params_dir(params_dir_str, pinyin_vec):
    group_name = params_dir_str.split('_', maxsplit=1)[0]
    lexicon_dict = 'ctb' if 'ctb' in params_dir_str else 'sgns'
    if re.findall('pinyin_\w+_freq', params_dir_str):
        fusion_pattern = 'freq'
    else:
        fusion_pattern = 'attn'
    return '_'.join([group_name, fusion_pattern, lexicon_dict, pinyin_vec])


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


def display_results(res_dict, target_dataset):
    table_list = []
    for dataset, param_dict in res_dict.items():
        if target_dataset != 'all' and target_dataset not in dataset:
            continue
        table = PrettyTable(['params', 'Precision', 'Recall', 'F1'])
        rows = resolve_data(param_dict)
        for row in rows:
            table.add_row(row)
        table_string = '='*100 + '\n' + dataset + '\n' + table.get_string()
        table_list.append(table_string)
    return table_list


def init_xlwt_style():
    font = xlwt.Font()
    font.name = 'Times New Roman'
    font.height = 20 * 11   # 字体大小，11为字号，20为衡量单位
    font.bold = False

    borders = xlwt.Borders()
    # 细实线:1，小粗实线:2，细虚线:3，中细虚线:4，大粗实线:5，双线:6，细点虚线:7
    # 大粗虚线:8，细点划线:9，粗点划线:10，细双点划线:11，粗双点划线:12，斜点划线:13
    borders.left = 1
    borders.right = 1
    borders.top = 1
    borders.bottom = 1

    alignment = xlwt.Alignment()
    # 0x01(左端对齐)、0x02(水平方向上居中对齐)、0x03(右端对齐)
    alignment.horz = 0x02
    # 0x00(上端对齐)、 0x01(垂直方向上居中对齐)、0x02(底端对齐)
    alignment.vert = 0x01

    style0 = xlwt.XFStyle()
    style0.font = font
    style0.borders = borders
    style0.alignment = alignment

    return style0


def set_font(font_name="Times New Roman", color=0, height=20*14, bold=False):
    font = xlwt.Font()  # 为样式创建字体
    font.name = font_name     # 字体类型：比如宋体、仿宋也可以是汉仪瘦金书繁
    font.colour_index = color     # 设置字体颜色
    font.height = height     # 字体大小
    font.bold = bold     # 粗体
    return font


def set_column_width(sheet):
    sheet.col(0).width = 100 * 256
    sheet.col(1).width = 16 * 256
    sheet.col(2).width = 16 * 256
    sheet.col(3).width = 16 * 256


def set_style(font_name="Times New Roman",
              height=20*14,
              color=0,
              borders_tags=None,
              bold=False):
    style = xlwt.XFStyle()  # 初始化样式
    font = set_font(font_name, color, height, bold)
    style.font = font
    # NO_LINE： 官方代码中NO_LINE所表示的值为0，没有边框; THIN： 官方代码中THIN所表示的值为1，边框为实线
    if borders_tags:
        borders = xlwt.Borders()
        borders.left, borders.right, borders.top, borders.bottom = borders_tags
        style.borders = borders
    alignment = xlwt.Alignment()
    # 0x01(左端对齐)、0x02(水平方向上居中对齐)、0x03(右端对齐)
    alignment.horz = 0x02
    # 0x00(上端对齐)、 0x01(垂直方向上居中对齐)、0x02(底端对齐)
    alignment.vert = 0x01
    style.alignment = alignment

    return style


def write_sheet_head(sheet, dataset, cur_row, hl_color=0x0a):
    sheet.write(cur_row, 0, dataset, set_style(color=hl_color, height=20*16, bold=False))
    column_list = ['Params', 'Precision', 'Recall', 'F1']
    for ith, v in enumerate(column_list):
        sheet.write(cur_row + 1, ith, v, set_style(borders_tags=[2, 2, 2, 2], bold=True))
    return cur_row + 2


def write_sheet_content(sheet, param_dict, cur_row, hl_color=0x0A):
    precision_list, recall_list, f1_list = [], [], []
    for _, score_dict in param_dict.items():
        precision_list.append(score_dict['Micro precision'])
        recall_list.append(score_dict['Micro recall'])
        f1_list.append(score_dict['Micro F1'])
    if len(precision_list) == 0 or len(recall_list) == 0 or len(f1_list) == 0:
        return cur_row
    precision_max = max(precision_list)
    recall_max = max(recall_list)
    f1_max = max(f1_list)

    for ith, (param_name, score_dict) in enumerate(param_dict.items()):
        if score_dict['pinyin_vec'] is None:
            continue
        else:
            span_micro_p = "%.2f" % score_dict['Span Micro precision']
            span_micro_r = "%.2f" % score_dict['Span Micro recall']
            span_micro_f1 = "%.2f" % score_dict['Span Micro F1']
            micro_p = "%.2f" % score_dict['Micro precision']
            micro_r = "%.2f" % score_dict['Micro recall']
            micro_f1 = "%.2f" % score_dict['Micro F1']
            micro_p_color = hl_color if score_dict['Micro precision'] == precision_max else 0
            micro_r_color = hl_color if score_dict['Micro recall'] == recall_max else 0
            micro_f1_color = hl_color if score_dict['Micro F1'] == f1_max else 0
            p_seg = [(span_micro_p, set_font()), (' / ', set_font()), (micro_p, set_font(color=micro_p_color))]
            r_seg = [(span_micro_r, set_font()), (' / ', set_font()), (micro_r, set_font(color=micro_r_color))]
            f1_seg = [(span_micro_f1, set_font()), (' / ', set_font()), (micro_f1, set_font(color=micro_f1_color))]
            bottom_tag = 2 if ith + 1 == len(param_dict) else 0

            simple_params = parse_params_dir(param_name, score_dict['pinyin_vec'])
            params_seg = [(simple_params, set_font()), (f"(dpr={score_dict['dpr']},wz={int(score_dict['wz'])})", set_font(color=hl_color))]
            sheet.write(cur_row, 0, params_seg, set_style(borders_tags=[2, 2, 0, bottom_tag]))
            sheet.write_rich_text(cur_row, 1, p_seg, set_style(borders_tags=[2, 2, 0, bottom_tag]))
            sheet.write_rich_text(cur_row, 2, r_seg, set_style(borders_tags=[2, 2, 0, bottom_tag]))
            sheet.write_rich_text(cur_row, 3, f1_seg, set_style(borders_tags=[2, 2, 0, bottom_tag]))
            cur_row += 1

    return cur_row


def save_as_excel(res_dict, save_path):
    workbook = xlwt.Workbook(encoding='utf-8')

    for dataset, param_dict in res_dict.items():
        cur_row = 0
        sheet = workbook.add_sheet(f"sheet_{dataset}")
        cur_row = write_sheet_head(sheet, dataset, cur_row)
        cur_row = write_sheet_content(sheet, param_dict, cur_row)
        set_column_width(sheet)
    workbook.save(save_path)


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
    # extract_score_from_logs(r'C:\NLP-Github\AIPolicy\output\entity\logs',
    #                         save_path='./example.xls')
    extract_score_from_logs('../output/entity/logs',
                            save_path='./entity.xls',
                            target_dataset='all')
