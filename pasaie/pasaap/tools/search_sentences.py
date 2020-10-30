import re
import os


def search_target_sentences(file_path):
    '''
        Search target sentences from raw text by rules.
    :param file_path: str, path of raw text
    :return:
    '''
    target_sentences = []
    chinese_numbers = list('一二三四五六七八九十')
    arabic_numbers = list(str(i) for i in range(10))

    possible_titles = [num + '、' for num in chinese_numbers] + ['第' + num for num in chinese_numbers]
    sub_titles = ["（{}）".format(num) for num in chinese_numbers]
    possible_title_content = ['申报要求', '申报条件', '申报范围', '基本要求', '基本条件', '满足以下条件', '企业条件', '总体要求', '申报资格要求', '类型和条件',
                              '推荐条件',
                              '举荐条件', '申报主体条件', '申报人条件', '机构条件', '选拔条件', '以下条件', '相关条件', '支持对象及条件', '举荐条件', '申请资格',
                              '申请对象', '申报资格', '支持对象', '申报对象']
    stable_expression = ['具备以下条件', '满足以下条件', '符合下列条件', '符合以下条件:', '符合以下条件：']
    exclusion = [num + '、其他' for num in arabic_numbers]

    is_target_span = False
    for line in open(file_path, 'r', encoding='utf8'):
        line = line.strip()
        if ((any([title in line[:10] for title in possible_titles + sub_titles]) and any(
                [content in line[:15] for content in possible_title_content])) or any(
                [exp in line for exp in stable_expression])) and all(not (s in line[:10]) for s in exclusion):
            if is_target_span:
                target_sentences.append('\n')
            is_target_span = True
        elif any([title in line[:10] for title in possible_titles]) or any(s in line[:10] for s in exclusion):
            if is_target_span:
                target_sentences.append('\n')
            is_target_span = False

        if is_target_span:
            target_sentences.append(line)

    return target_sentences


def accuracy_of_pattern_match(src_dir, tgt_dir, exclusion_path):
    src_files_num = len([file for file in os.listdir(src_dir) if file.endswith('.txt')])
    tgt_files_num = len([file for file in os.listdir(tgt_dir) if file.endswith('.txt')])
    exclusion_num = len([line for line in open(exclusion_path, 'r', encoding='utf8')])
    acc = (tgt_files_num + exclusion_num) / src_files_num
    return acc


def cut_sent(sent):
    sent = re.sub('([。！？\?])([^”’])', r"\1\n\2", sent)  # 单字符断句符
    sent = re.sub('(\.{6})([^”’])', r"\1\n\2", sent)  # 英文省略号
    sent = re.sub('(\…{2})([^”’])', r"\1\n\2", sent)  # 中文省略号
    sent = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', sent)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    sent = sent.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return sent.split("\n")


def simple_sentence_filter(sent):
    """
        Return False when the `sent` is unnecessary.
    :param sent: str, input sentence
    :return:
    """
    start_expression = ['重点支持', '积极支持']
    unnecessary_expression = ['有较强的', '具有良好', '诚信状况良好', '重复申报', '多头申报', '项目才可正式立项', '详见', '原则上']
    if any(sent.startswith(s) for s in start_expression) or any(s in sent for s in unnecessary_expression):
        return False
    else:
        return True


def num_of_article_contains_specific_str(src_dir, specific_regex):
    """
        Count the number and ratio of files contain specific regex.
    :param src_dir: str, source directory contains many txt files
    :param specific_regex: str, specific regex to search
    :return:
        tuple, number of files contain specific regex ard Proportion
    """
    total_num = 0
    specific_num = 0
    specific_list = []
    for file in os.listdir(src_dir):
        filepath = os.path.join(src_dir, file)
        if filepath.endswith('.txt'):
            total_num += 1
            with open(filepath, 'r', encoding='utf8') as fin:
                for line in fin:
                    if re.search(specific_regex, line.strip()):
                        specific_num += 1
                        specific_list.append(file)
                        break
    return specific_num, specific_num / total_num, specific_list


if __name__ == '__main__':
    # search_acc = accuracy_of_pattern_match(src_dir=r'C:\Users\90584\Desktop\政策实体与关系抽取\语料\clean-jiangbei1',
    #                                 tgt_dir=r'C:\NLP-Github\PolicyMining\article_parsing\output\target',
    #                                 exclusion_path=r'C:\NLP-Github\PolicyMining\article_parsing\output\no_content.txt')
    # print("Sentence search accuracy: {}".format(search_acc))

    specific_num, ratio, specific_list = num_of_article_contains_specific_str(
        src_dir=r'C:\Users\90584\Desktop\政策实体与关系抽取\语料\clean-jiangbei1',
        specific_regex='(条件之一)|(下列.*之一)')
    print(specific_num, '\n', ratio, '\n', specific_list)
