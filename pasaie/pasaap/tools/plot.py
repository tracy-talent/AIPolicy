#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2020/10/26 16:27
# @Author:  Mecthew
import os
try:
    from graphviz import Digraph
except Exception as e:
    print(f'\033[31mException occur: {e}')
    print('\033[0m')
node_idx = "0"
cur_dir = os.path.dirname(__file__)


def plot_tree(tree, name, output_dir=None, max_display_len=50, view=False):
    global node_idx
    name = name.split('.')[0]

    out_file = os.path.join(output_dir, name)
    os.makedirs(output_dir, exist_ok=True)
    g = Digraph("G", filename=out_file, format='png', strict=False)
    first_label = list(tree.keys())[0]
    node_content = refactor_content(first_label, max_display_len)
    g.node("0", node_content, fontname="Microsoft YaHei")
    _sub_plot_tree(g, tree, "0", max_display_len)
    g.render(view=view, cleanup=True)


def _sub_plot_tree(g, tree, inc, max_display_len):
    global node_idx

    first_label = list(tree.keys())[0]
    ts = tree[first_label]
    for i in ts.keys():
        if isinstance(tree[first_label][i], dict):
            node_idx = str(int(node_idx) + 1)
            content = refactor_content(list(tree[first_label][i].keys())[0], max_display_len)
            g.node(node_idx, content, fontname="Microsoft YaHei")
            g.edge(inc, node_idx, str(i), fontname="Microsoft YaHei")
            _sub_plot_tree(g, tree[first_label][i], node_idx, max_display_len)
        else:
            node_idx = str(int(node_idx) + 1)
            content = refactor_content(tree[first_label][i], max_display_len)
            g.node(node_idx, content, fontname="Microsoft YaHei")
            g.edge(inc, node_idx, str(i), fontname="Microsoft YaHei")


# The following codes are just used to get better display
def refactor_content(content, max_display_len):
    if len(content) > max_display_len:
        content = content[:max_display_len]
        suffix = '...'
    else:
        suffix = ''

    unit_num = 10
    if len(content) > unit_num:
        cnt_idx = 1
        new_content = ''
        while cnt_idx * unit_num < len(content):
            new_content += content[unit_num * (cnt_idx - 1): unit_num * cnt_idx] + '\n'
            cnt_idx += 1
        new_content += content[unit_num * (cnt_idx - 1):]
        content = new_content + suffix
    return content
