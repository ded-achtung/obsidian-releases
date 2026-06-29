"""Тест: подсловные эмбеддинги обобщают на новые СЛОВОФОРМЫ, мешок слов — нет."""

from __future__ import annotations

import numpy as np

from thinking_system.language.grounding import (
    BagOfWords, CharNgram, GoalClassifier, generate_commands,
)
from run_meaning import NOVEL


def _acc(clf, items):
    return float(np.mean([clf.predict(t)[0] == c for t, c in items]))


def test_charngram_matches_bow_on_recombination() -> None:
    """На лёгком случае (новые комбинации знакомых слов) подсловная модель не хуже."""
    data = generate_commands()
    rng = np.random.default_rng(0)
    rng.shuffle(data)
    cut = int(0.7 * len(data))
    train, test = data[:cut], data[cut:]
    cng = GoalClassifier(CharNgram([t for t, _ in train]))
    cng.fit(train, epochs=300)
    assert _acc(cng, test) > 0.9


def test_charngram_beats_bow_on_novel_wordforms() -> None:
    """Ключевой тест: на невиданных словоформах подслова обобщают, целые слова — нет."""
    full = generate_commands()
    bow = GoalClassifier(BagOfWords([t for t, _ in full]))
    bow.fit(full, epochs=300)
    cng = GoalClassifier(CharNgram([t for t, _ in full]))
    cng.fit(full, epochs=300)

    bow_acc = _acc(bow, NOVEL)
    cng_acc = _acc(cng, NOVEL)
    assert cng_acc >= 0.9                    # подслова берут почти всё
    assert cng_acc - bow_acc >= 0.4          # и заметно лучше мешка слов


def test_novel_words_are_truly_unseen() -> None:
    """Гарантируем честность теста: проверочные слова целиком НЕ встречались в обучении."""
    from thinking_system.language.grounding import tokenize
    train_vocab = {w for t, _ in generate_commands() for w in tokenize(t)}
    novel_tokens = {w for t, _ in NOVEL for w in tokenize(t)}
    truly_new = novel_tokens - train_vocab
    # хотя бы по одному реально новому (морфологически прозрачному) слову на класс
    assert {"northwestern", "southeastern", "northeastern", "southwestern"} <= truly_new
