# -*-coding:utf-8-*-
"""
@Time    : 2021/7/12 0:52
@Author  : Mecthew
@File    : pinyin_statistic.py
"""
import time
import requests
import json
from pypinyin import pinyin
import lxml.etree as etree
from pasaie.utils.entity_extract import extract_kvpairs_in_bmoes


def count_pinyin_in_nerfile(filepath, use_xinhua=True):
    total_entities = []
    with open(filepath, 'r', encoding='utf8') as fin:
        text, labels = [], []
        for line in fin:
            if line.strip():
                token, label = line.strip().split()
                text.append(token)
                labels.append(label)
            else:
                entities = extract_kvpairs_in_bmoes(labels, text)
                total_entities.extend(entities)
                text, labels = [], []
    boundary_polyphonic_entities = []
    entity_polyphonic_entities = []
    word2pinyin = json.load(open('word2pinyin.json', encoding='utf8'))
    for entity in total_entities:
        entity_name = entity[-1]
        if use_xinhua:
            if len(get_pinyin(word2pinyin, entity_name[0])) > 1 or len(get_pinyin(word2pinyin, entity_name[-1])) > 1:
                boundary_polyphonic_entities.append(entity)
            for zh_ch in entity_name:
                if len(get_pinyin(word2pinyin, zh_ch)) > 1:
                    entity_polyphonic_entities.append(entity)
                    break
        else:
            if any([l > 1 for l in map(len, pinyin(entity_name[0] + entity_name[-1], heteronym=True))]):
                boundary_polyphonic_entities.append(entity)
            if any([l > 1 for l in map(len, pinyin(entity_name, heteronym=True))]):
                entity_polyphonic_entities.append(entity)

    print(filepath)
    print(boundary_polyphonic_entities[-2:], entity_polyphonic_entities[-2:])

    result = {'total_entities': len(total_entities),
              'boundary_polyphonic_entities': len(boundary_polyphonic_entities),
              'boundary_polyphonic_entities_rate': round(len(boundary_polyphonic_entities) / len(total_entities) * 100,
                                                         3),
              'entity_polyphonic_entities': len(entity_polyphonic_entities),
              'entity_polyphonic_entities_rate': round(len(entity_polyphonic_entities) / len(total_entities) * 100, 3)}
    return result


def count_dataset_pinyin(dataset, use_xinhua=True):
    suffix = 'clip256.' if dataset in ['msra', 'ontonotes4'] else ''
    if dataset == 'msra':
        file_dict = {'train': f'../../input/benchmark/entity/{dataset}/train.char.{suffix}bmoes',
                     'test': f'../../input/benchmark/entity/{dataset}/test.char.{suffix}bmoes'}
    else:
        file_dict = {'train': f'../../input/benchmark/entity/{dataset}/train.char.{suffix}bmoes',
                     'dev': f'../../input/benchmark/entity/{dataset}/dev.char.{suffix}bmoes',
                     'test': f'../../input/benchmark/entity/{dataset}/test.char.{suffix}bmoes'}
    output_dict = dict()
    suffix = "_xinhua" if use_xinhua else '_pypinyin'
    output_path = f'./{dataset}_pinyin_statistics{suffix}.json'
    fout = open(output_path, 'w', encoding='utf8')
    total_entities_len, total_boundary_len, total_entities_poly_len = 0, 0, 0
    for k, v in file_dict.items():
        output_dict[k] = count_pinyin_in_nerfile(v, use_xinhua=use_xinhua)
        total_entities_len += output_dict[k]['total_entities']
        total_boundary_len += output_dict[k]['boundary_polyphonic_entities']
        total_entities_poly_len += output_dict[k]['entity_polyphonic_entities']
    output_dict['all_entities'] = total_entities_len
    output_dict['all_boundary_polyphonic_rate'] = round(total_boundary_len / total_entities_len * 100, 3)
    output_dict['all_entity_polyphonic_rate'] = round(total_entities_poly_len / total_entities_len * 100, 3)
    json.dump(output_dict, fout, ensure_ascii=False, indent=0)


def get_xinhua_zh2pinyin():
    word2pinyin = dict()
    missed_chars = dict()

    def get_page(zh_char, sleep_time=0.1):
        base_url = 'https://zd.hwxnet.com/search.do?keyword={}&sub_btn.x=0&sub_btn.y=0'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE'
        }
        time.sleep(sleep_time)
        try:
            response = requests.get(base_url.replace('{}', zh_char), headers=headers)
            response.encoding = 'utf-8'
            selector = etree.HTML(response.content)
            pinyin_list = selector.xpath(
                "//div[@id='container']/div[@id='intro']/div[@id='sub_con']/div[@class='introduce']/div[@class='label'][2]/span[@class='pinyin']/text()")
            pinyin_list = [p for p in pinyin_list if p.strip()]
            word2pinyin[zh_char] = pinyin_list
        except Exception as e:
            print(e)
            missed_chars[zh_char] = len(missed_chars)

    for idx in range(0x4e00, 0x9fa5 + 1):
        utf8_str = chr(idx)
        get_page(utf8_str)
    with open('word2pinyin.json', 'w', encoding='utf8') as fout1:
        json.dump(word2pinyin, fout1, ensure_ascii=False, indent=0)
    with open('./missed_chars.json', 'w', encoding='utf8') as fout2:
        json.dump(missed_chars, fout2, ensure_ascii=False, indent=0)


def get_pinyin(word2pinyin, zh_char):
    if zh_char in word2pinyin:
        return word2pinyin[zh_char]
    else:
        return [zh_char]


if __name__ == '__main__':
    dataset = 'weibo'
    suffix = '' if dataset in ['weibo', 'resume'] else 'clip256.'
    # count_pinyin_in_nerfile(filepath)

    # get_xinhua_zh2pinyin()

    for dataset in ['msra', 'ontonotes4', 'weibo', 'resume']:
        count_dataset_pinyin(dataset, use_xinhua=True)

    # word2pinyin = json.load(open('word2pinyin.json', encoding='utf8'))
    # print(get_pinyin(word2pinyin, '昊'))