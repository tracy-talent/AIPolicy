import re
import unicodedata
from ..tokenization.utils import strip_accents


def is_eng_word(word):
    word = strip_accents(word)
    if re.match('^[a-zA-Z]+(-[a-zA-Z]+)?(\'?[a-zA-Z]*)$', word):
        return True
    else:
        return False


def is_digit(word):
    if re.match('^[+-]?\d+(\.\d+)?$', word):
        return True
    else:
        return False


def is_pinyin(word):
    if re.match('^[a-z]{1,6}[1-5]$', word):
        return True
    else:
        return False


def is_punctuation(word):
    if word in ['￥', '%', '：', '—', '-', '&', '。', '…', '’', '？', '！', '”', '#', '(', '[', ']', '）', '、', '~',
                '!', ')', '@', '{', '|', '*', '\\', '`', '?', '·', '.', '_', '>', '》', '】', '；', '$', '（', ':',
                '/', '“', '+', '}', "'", '"', '，', '【', '《', ';', ',', '=', '^', '<', '‘']:
        return True
    else:
        return False


# 由于英文单词/数字会被lazy_pinyin作为一个整体作为元素，所以需要展开为char，与word长度一致
def unfold_pinyin_list(pinyin_list):
    unfold_list = []
    for pinyin in pinyin_list:
        if is_pinyin(pinyin):
            unfold_list.append(pinyin)
        else:
            unfold_list += list(pinyin)
    return unfold_list
