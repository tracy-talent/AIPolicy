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
    if re.match('^[a-z]{1,6}+[1-5]$', word):
        return True
    else:
        return False
