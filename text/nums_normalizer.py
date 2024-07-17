import re
from .kazakh_text_utils import kazakh_numerals_ordinal_mapping
from num2words import num2words


def _replace_nums_pair_word(match):
    nums, word = match.group().split()
    return "{} {}".format(__get_ordinal(nums), word)


def __get_ordinal(nums):
    wordnums = num2words(nums, lang="kz").rsplit(" ", 1)
    wordnums[-1] = kazakh_numerals_ordinal_mapping[wordnums[-1]]
    return " ".join(wordnums)


def _replace_nums(match):
    return num2words(match.group(), lang="kz")


def _remove_nums(match):
    return re.sub('\d', '', match.group())


def _replace_ordinal_nums(match):
    nums, suffix = match.group().split("-")
    return __get_ordinal(nums)


def _replace_group_nums(match):
    nums, suffix = match.group().split("-")
    nums = num2words(nums, lang="kz")+suffix
    return nums
