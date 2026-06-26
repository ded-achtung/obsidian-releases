"""Подготовка данных и оценка для символьного предсказания.

Главная честная метрика — bits-per-character (BPC) на ОТЛОЖЕННОЙ части текста,
сравниваемая с тривиальными бейзлайнами:
  • uniform — равномерное угадывание: log2(размер словаря);
  • unigram — по частотам символов обучающей части (без учёта контекста).
Модель «выучила структуру», если её held-out BPC заметно ниже обоих.
"""

from __future__ import annotations

import numpy as np


def make_pairs(ids: np.ndarray, context_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Скользящее окно: (контексты (M, context_len), целевые символы (M,))."""
    n = len(ids) - context_len
    if n <= 0:
        return np.empty((0, context_len), dtype=np.int64), np.empty((0,), dtype=np.int64)
    contexts = np.stack([ids[i : i + context_len] for i in range(n)])
    targets = ids[context_len:]
    return contexts, targets


def train_test_split(ids: np.ndarray, *, train_frac: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Разбить последовательность по позиции (без перемешивания — текст непрерывный)."""
    cut = int(len(ids) * train_frac)
    return ids[:cut], ids[cut:]


def uniform_bpc(vocab_size: int) -> float:
    """BPC равномерного бейзлайна."""
    return float(np.log2(vocab_size))


def unigram_bpc(train_ids: np.ndarray, test_targets: np.ndarray, vocab_size: int) -> float:
    """BPC по частотам символов обучающей части (со сглаживанием Лапласа)."""
    counts = np.bincount(train_ids, minlength=vocab_size).astype(np.float64) + 1.0
    p = counts / counts.sum()
    return float(np.mean(-np.log2(p[test_targets])))


def eval_bpc(predictor, contexts: np.ndarray, targets: np.ndarray, *, batch: int = 1024) -> float:
    """Средний BPC модели на наборе (по батчам, без обучения)."""
    if len(contexts) == 0:
        return float("nan")
    total = 0.0
    for i in range(0, len(contexts), batch):
        total += predictor.nll_bits(contexts[i : i + batch], targets[i : i + batch]) * len(contexts[i : i + batch])
    return total / len(contexts)


def accuracy(predictor, contexts: np.ndarray, targets: np.ndarray, *, batch: int = 1024) -> float:
    """Доля верно угаданного следующего символа (argmax)."""
    if len(contexts) == 0:
        return float("nan")
    correct = 0
    for i in range(0, len(contexts), batch):
        logits, _ = predictor._forward(contexts[i : i + batch])
        correct += int(np.sum(logits.argmax(axis=1) == targets[i : i + batch]))
    return correct / len(contexts)
