"""Текстовый домен: ингест книг (PDF/MD/txt) → символы → обучаемый поток.

Подключает РЕАЛЬНЫЕ данные (книги) к системе через тот же принцип «предсказывай
следующее и учись на ошибке». Качество — bits-per-character на отложенной части.
"""

from thinking_system.text.ingest import load_book, load_corpus
from thinking_system.text.vocab import CharVocab
from thinking_system.text.dataset import (
    accuracy,
    eval_bpc,
    make_pairs,
    train_test_split,
    unigram_bpc,
    uniform_bpc,
)

__all__ = [
    "load_book",
    "load_corpus",
    "CharVocab",
    "make_pairs",
    "train_test_split",
    "uniform_bpc",
    "unigram_bpc",
    "eval_bpc",
    "accuracy",
]
