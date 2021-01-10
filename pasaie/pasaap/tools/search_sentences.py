import configparser
import os
import re
import sys

sys.path.append('../../..')
from pasaie.pasaap.tools import LogicTree, LogicNode

project_path = '/'.join(os.path.abspath(__file__).replace('\\', '/').split('/')[:-4])
config = configparser.ConfigParser()
config.read(os.path.join(project_path, 'config.ini'))

subtitle_pattern1 = re.compile('^第[一二三四五六七八九十]+章')
subtitle_pattern2 = re.compile('^第[一二三四五六七八九十]+条')
subtitle_pattern3 = re.compile('^[一二三四五六七八九十]+(、|\.)')
subtitle_pattern4 = re.compile('^(\(|（)[一二三四五六七八九十]+(\)|）)')
subtitle_pattern5 = re.compile('^[0-9]+(\.|．|、)')
subtitle_pattern6 = re.compile('^(\(|（)[0-9]+(\)|）)')
subtitle_pattern7 = re.compile('^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]+')
subtitle_pattern7_1 = re.compile('[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]+')
subtitle_pattern_dict = {
    '章': subtitle_pattern1,
    '条': subtitle_pattern2,
    '一、': subtitle_pattern3,
    '(一)': subtitle_pattern4,
    '1.': subtitle_pattern5,
    '(1)': subtitle_pattern6,
    '①': subtitle_pattern7
}
possible_title_content = ['申报要求', '申报条件', '申报范围', '基本要求', '基本条件', '满足以下条件', '企业条件', '总体要求', '申报资格要求', '类型和条件', '资格条件',
                          '举荐条件', '申报主体条件', '申报人条件', '机构条件', '选拔条件', '以下条件', '相关条件', '支持对象及条件', '举荐条件', '申请资格',
                          '申请对象', '申报资格', '支持对象', '申报对象', '推荐条件', '提名要求', '资助对象', '资助范围', '支持标准', '遴选条件', '参评主体条件']
stable_expression = ['具备以下条件', '满足以下条件', '符合下列条件', '符合以下条件', '符合下列条件', '满足以下标准', '条件之一']
regex_exp = ['申请.*的机构', '本细则适用于.*(企业|单位)']
sentence_level_or_exp = ['条件.*之一', '任意。*条件', '条件.*一项', '分别.*满足', '分类.*条件',
                         '中.*选择一']


def get_article_ladder(file_path):
    article = open(file_path, 'r', encoding='utf8').readlines()
    article_ladder = []
    prev_subtitle = ''
    # construct title ladder
    for line in article:
        content = line.strip()
        for kstr, pattern in subtitle_pattern_dict.items():
            matched = pattern.search(content)
            if matched:
                if kstr not in article_ladder:
                    prev_idx = article_ladder.index(prev_subtitle) if prev_subtitle else 0
                    article_ladder.insert(prev_idx + 1, kstr)
                prev_subtitle = kstr
                break

    return article_ladder


def get_article_structure(file_path):
    article_ladder = get_article_ladder(file_path)
    article = open(file_path, 'r', encoding='utf8').readlines()
    title_start_flag = True
    title_start_content = []
    root_node = None
    prev_node_idx, prev_node = -1, None
    for lidx in range(len(article)):
        content = article[lidx].strip().replace(u'\xa0', ' ')
        if content == '':
            continue
        is_matched = False
        for kidx, kstr in enumerate(article_ladder):
            matched = subtitle_pattern_dict[kstr].search(content)
            if matched:
                matched_subtitle = matched.group(0)
                content = content[len(matched_subtitle):]
                if title_start_flag:
                    root = LogicNode(is_root=True, logic_type='AND', sentence='\n'.join(title_start_content))
                    root_node, prev_node = root, root

                child_node = LogicNode(is_root=False, logic_type='AND', sentence=content)
                if kidx >= prev_node_idx + 1:
                    prev_node.add_node(child_node, node_key=matched_subtitle)
                    prev_node.is_root = True
                    prev_node.logic_type = judge_node_logic_type(prev_node)
                else:
                    tmp_idx = prev_node_idx
                    while tmp_idx + 1 > kidx:
                        prev_node = prev_node.parent
                        tmp_idx = prev_node.depth - 1
                    prev_node.add_node(child_node, node_key=matched_subtitle)
                    prev_node.is_root = True
                    prev_node.logic_type = judge_node_logic_type(prev_node)

                # 目的是为了分割句子中的①
                if kidx == len(article_ladder) - 1 and subtitle_pattern7_1.findall(content):
                    tail_subtitle = subtitle_pattern7_1.finditer(content)
                    content_idx = 0
                    child_node.convert_to_root_node(logic_type='AND', sentence=child_node.get_sentence())
                    for tspan in tail_subtitle:
                        pos_b, pos_e = tspan.span()[0], tspan.span()[1]
                        if content_idx == 0:
                            child_node.sent = content[content_idx: pos_b]
                        else:
                            tmp_node = LogicNode(is_root=False, logic_type=None, sentence=content[content_idx: pos_b])
                            child_node.add_node(tmp_node, node_key=content[content_idx: content_idx + 1])
                        content_idx = pos_b
                    if content_idx < len(content):
                        tmp_node = LogicNode(is_root=False, logic_type=None, sentence=content[content_idx:])
                        child_node.add_node(tmp_node, node_key=content[content_idx: content_idx + 1])
                    child_node.logic_type = judge_node_logic_type(child_node)
                prev_node = child_node
                prev_node_idx = kidx
                is_matched = True
                title_start_flag = False
                break

        if not is_matched and not title_start_flag:
            if prev_node_idx == len(article_ladder) - 1 and prev_node.is_root is False:
                # 最底层小标题是否继续细分
                if any(exp in prev_node.sent[-15:] for exp in ['如下', '下述', '下列', '下面', '以下', '之一']) or \
                        len(prev_node.sent) < 15:
                    prev_node_logic_type = judge_node_logic_type(prev_node)
                    prev_node.convert_to_root_node(logic_type=prev_node_logic_type,
                                                   sentence=prev_node.get_sentence())
                    content_node = LogicNode(is_root=False, logic_type=None, sentence=content)
                    prev_node.add_node(content_node)
                else:
                    prev_node.sent += '\n' + content
            elif len(prev_node.children) == 0:
                prev_node.sent += '\n' + content
            else:
                prev_node.logic_type = judge_node_logic_type(prev_node)
                prev_node.add_node(node=LogicNode(is_root=False, logic_type=None, sentence=content))
        if title_start_flag:
            title_start_content.append(content)

    if root_node is None:
        return None
    tree = LogicTree(root=root_node, json_path=None)
    output_dir = os.path.join(config['path']['output'], 'article_parsing/raw-policy/article_structure')
    os.makedirs(output_dir, exist_ok=True)
    tree.save_as_json(output_path=os.path.join(output_dir, file_path.split('/')[-1].replace('.txt', '.json')))
    try:
        tree.save_as_png(output_dir=output_dir, filename=file_path.split('/')[-1])
    except:
        pass
    return tree


def judge_node_logic_type(node):
    if any([re.findall(exp, node.get_sentence()) for exp in sentence_level_or_exp]):
        return 'OR'
    else:
        return 'AND'


def judge_sent_logic_type(sent):
    if any([re.findall(exp, sent) for exp in sentence_level_or_exp]):
        return 'OR'
    else:
        return 'AND'


def get_target_tree(file_path, output_dir=None, require_json=True, require_png=True):
    if output_dir is None:
        output_dir = os.path.join(config['path']['output'], 'article_parsing/raw-policy/pruning_tree')

    article_structure_tree = get_article_structure(file_path)
    if article_structure_tree is None:
        return None
    retain_list = []
    recursive_prune_tree(article_structure_tree.root, retain_list)
    recheck_tree(article_structure_tree.root)

    try:
        if require_json:
            os.makedirs(output_dir, exist_ok=True)
            article_structure_tree.save_as_json(os.path.join(output_dir, file_path.split('/')[-1].replace('.txt', '.json')))
    except Exception as e:
        print(f"{file_path.replace('.txt', ''): {e}}")
    try:
        if require_png:
            os.makedirs(output_dir, exist_ok=True)
            article_structure_tree.save_as_png(output_dir, file_path.split('/')[-1].replace('.txt', '.png'))
    except Exception as e:
        print(e)

    return article_structure_tree


def recursive_prune_tree(root_node, retain_list):
    if judge_whether_to_retain(root_node.get_sentence()):
        retain_list.append(id(root_node))
    else:
        del_keys = []
        for child_name, child_node in root_node.children.items():
            recursive_prune_tree(child_node, retain_list)
            if id(child_node) not in retain_list:
                del_keys.append(child_name)
        for key in del_keys:
            if key in root_node.children:
                del root_node.children[key]

        # if len(root_node.children) == 1:        # 叶节点上移
        #     retain_list.append(id(root_node))
        if len(root_node.children) > 0:
            retain_list.append(id(root_node))


def recheck_tree(root_node):
    sent = root_node.get_sentence()
    split_str = re.split('[。?？！!\n]', sent)
    split_str = [sstr for sstr in split_str if sstr.strip() and simple_sentence_filter(sstr)]
    if root_node.is_root:
        del_names = []
        for child_name, child_node in root_node.children.items():
            if not recheck_tree(child_node):
                del_names.append(child_name)
        for name in del_names:
            del root_node.children[name]
        if len(split_str) > 0:
            logic_type = 'AND'
            for sstr in split_str:
                if any(re.findall(pattern, sstr) for pattern in sentence_level_or_exp):
                    logic_type = 'OR'
                    break
            root_node.logic_type = logic_type
            if root_node.depth > 0:  # 非根节点执行以下操作
                root_node.sent = split_str[0]
                # FIXME: 这里把root_node标题也加入到
                split_str = split_str if len(split_str[0]) > 20 else split_str[1:]
                for sstr in split_str:
                    root_node.add_node(LogicNode(is_root=False, logic_type=None, sentence=sstr))
        return len(root_node.children) > 0
    else:
        if len(split_str) == 1:  # 由于sentence_filter的存在，一些句子可能被过滤
            root_node.sent = split_str[0]
        elif len(split_str) > 1:
            logic_type = 'AND'
            for sstr in split_str:
                if any(re.findall(pattern, sstr) for pattern in sentence_level_or_exp):
                    logic_type = 'OR'
                    break
            root_node.convert_to_root_node(logic_type=logic_type, sentence=split_str[0])
            split_str = split_str if len(split_str[0]) > 20 else split_str[1:]
            for sstr in split_str:
                root_node.add_node(LogicNode(is_root=False, logic_type=None, sentence=sstr))
        return len(split_str) > 0


def judge_whether_to_retain(sentence):
    if not sentence:
        return False
    elif any(exp in sentence[:20] for exp in possible_title_content) or \
            any(exp in sentence for exp in stable_expression) or \
            any(re.match(pattern, sentence) for pattern in regex_exp):
        return True
    else:
        return False


def accuracy_of_pattern_match(src_dir, tgt_dir, exclusion_path):
    src_files_num = len([file for file in os.listdir(src_dir) if file.endswith('.txt')])
    tgt_files_num = 0
    no_content_files = []
    for file in os.listdir(tgt_dir):
        if file.endswith('.txt'):
            file_path = os.path.join(tgt_dir, file)
            with open(file_path, encoding='utf8') as fin:
                article = ''.join([line.strip() for line in fin])
                if article:
                    tgt_files_num += 1
                else:
                    no_content_files.append(file)
    exclusion_files = [line.strip() for line in open(exclusion_path, 'r', encoding='utf8') if line.strip()]
    no_content_files = list(set(no_content_files) - set(exclusion_files))
    exclusion_num = len(
        [line for line in open(exclusion_path, 'r', encoding='utf8') if line.strip()])  # mismatched_files
    acc = (tgt_files_num + exclusion_num) / src_files_num
    print(f'Accuracy of files extracted target sentences: {acc}')
    print(f"empty matched files: \n{sorted(no_content_files, key=lambda x: x.replace('.txt', ''))}")
    # print(f"exclusion files: \n{sorted(exclusion_files, key=lambda x: int(x.replace('.txt', '')))}")
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
    print(specific_list)
    return specific_num, specific_num / total_num, specific_list


if __name__ == '__main__':
    src_dir = config['path']['input'] + '/benchmark/article_parsing/raw-policy'
    tgt_dir = config['path']['output'] + '/article_parsing/raw-policy/target_sentences/'
    exclusion_file_path = config['path']['input'] + '/benchmark/article_parsing/no_content_files.txt'

    for file in os.listdir(src_dir):
        if file.endswith('.txt'):
            file_path = os.path.join(src_dir, file)
            file_path = file_path.replace('\\', '/')
            get_target_tree(file_path)

