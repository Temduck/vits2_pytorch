""" from https://github.com/keithito/tacotron """

"""
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
"""

import re
from unidecode import unidecode
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
import nums_normalizer
backend = EspeakBackend("en-us", preserve_punctuation=True, with_stress=True)
backend_kz = EspeakBackend("kk", preserve_punctuation=True, with_stress=True)


# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]

_numerals = [
    (re.compile(x[0]), x[1])
    for x in [
        (r'\b(3[01]|[12][0-9]|[1-9])\s(қаңтар|ақпан|наурыз|сәуір|мамыр|маусым|шілде|тамыз|қыркүйек|қазан|қараша|желтоқсан)', '_replace_nums_pair_word'), # nums pair kazakh month
        (r'\b\d{4}\s(жыл)', '_replace_nums_pair_word'), # nums pair kazakh year        
        (r'\b\d{1,3}\b', '_replace_nums'), # hundreds 
        (r'[а-яА-ЯӘәҒғҚқҢңӨөҰұҮүҺһІі]+\d+', '_remove_nums'), # kazakh word with digit
        (r'\d+[а-яА-ЯӘәҒғҚқҢңӨөҰұҮүҺһІі]+', '_remove_nums'), # digit with kazakh word 
        (r'\d+-[інші|ыншы|сыншы|ші|шы]', '_replace_ordinal_nums'),  # ordianal numerals with suffix 
        (r'\d+-(ден|тан|тен)', '_replace_group_nums') # group numerals with suffix 
    ]
]

_issaitts_trash = [
    ((re.compile("%s" % x[0], re.IGNORECASE), x[1]))
    for x in [
        ('–|—|−|－', '-'),
        ("\n|noise|ʨ|ɕ|»|–|«|—|̆|“|”|…|−|－|●", '')
    ]
]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    for regex, replacement_func_name in _numerals:
        replacement_func = getattr(nums_normalizer, replacement_func_name)
        text = regex.sub(replacement_func, text)
    return text


def remove_trash(text, trash=_issaitts_trash):
    for regex, replacement in trash:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    """Pipeline for English text, including abbreviation expansion."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = phonemize(text, language="en-us", backend="espeak", strip=True)
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def english_cleaners2(text):
    """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = phonemize(
        text,
        language="en-us",
        backend="espeak",
        strip=True,
        preserve_punctuation=True,
        with_stress=True,
    )
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def english_cleaners3(text):
    """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = backend.phonemize([text], strip=True)[0]
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def kazakh_cleaners_issaitts(text):
    """Pipeline for Kazakh tts speakers datasets text, including num2words, + punctuation + g2p"""
    text = lowercase(text)
    text = expand_numbers(text)
    text = remove_trash(text, _issaitts_trash)
    phonemes = backend_kz.phonemize([text], strip=True)[0]
    phonemes = collapse_whitespace(phonemes)
    return phonemes