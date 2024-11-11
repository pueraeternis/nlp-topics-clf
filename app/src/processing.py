import re

import pymorphy3
from stop_words import get_stop_words

morph = pymorphy3.MorphAnalyzer()
stopwords = set(get_stop_words("ru"))


def standardize_text(text: str) -> str:
    special_symbol_pattern = re.compile(r"[^\w\s\-а-яА-Я]+", flags=re.UNICODE)
    text = special_symbol_pattern.sub("", text)
    text = re.sub(r"ё", "е", text)
    text = re.sub(r"[0-9]", "", text)
    text = re.sub(r"-", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip().lower()
    return text


def normalize_text(text: str) -> str:
    tokens = re.findall(r"\b\w+\b", text)
    lemmas = [
        get_normal_form(token)
        for token in tokens
        if len(token) > 1 and token not in stopwords
    ]
    return " ".join(lemmas)


def get_normal_form(token: str) -> str:
    return morph.parse(token)[0].normal_form  # type: ignore


def process_text(text: str) -> str:
    if isinstance(text, str):
        text_standardized = standardize_text(text)
        text_normalized = normalize_text(text_standardized)
        return text_normalized
    else:
        return " "
