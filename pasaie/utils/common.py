import re
import unicodedata


def is_eng_word(word):
    word = strip_accents(word)
    if not re.sub('[a-zA-Z]+(-[a-zA-Z]+)?(\'?[a-zA-Z]*)', '', word):
        return True
    else:
        return False


def is_digit(word):
    if not re.sub('([+-]?\d+)(\.\d+)?', '', word):
        return True
    else:
        return False


# 去掉类似café的音调
def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


