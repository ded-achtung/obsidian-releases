#!/usr/bin/env python3
"""Смысл вместо ключевых слов: подсловные эмбеддинги обобщают на НОВЫЕ словоформы.

Разрыв, который чинит этот скрипт: прежний грунтинг (мешок ЦЕЛЫХ слов) понимал лишь
новые КОМБИНАЦИИ знакомых слов — невиданное слово он молча отбрасывал. Подсловный
фичеризатор (char-n-граммы, как fastText, но без предобучения) представляет слово
его частями, поэтому «northwestern» узнаётся через общие n-граммы с «north»/«west».

Честный тест — НОВЫЕ СЛОВОФОРМЫ, которых целиком не было в обучении (а не просто
перестановки знакомых слов). Бейзлайн (мешок слов) на них падает, подсловная модель —
обобщает. Это «смысл как близость по форме», выученная под задачу, — не семантика в
полном смысле, но честный шаг от точного совпадения строк к обобщению на слово.

Запуск: python run_meaning.py
"""

from __future__ import annotations

import numpy as np

from thinking_system.language.grounding import (
    BagOfWords, CharNgram, GoalClassifier, PLACES, generate_commands,
)

# НОВЫЕ словоформы: ни одно из этих ключевых слов не встречается ЦЕЛИКОМ в обучении,
# но морфологически прозрачно (делит подслова с знакомыми направлениями).
NOVEL = [
    ("go to the northwestern corner", 0),     # northwestern ⊃ north, west
    ("head to the southeastern corner", 3),    # southeastern ⊃ south, east
    ("move to the northeastern edge", 1),      # northeastern ⊃ north, east
    ("navigate to the southwestern area", 2),  # southwestern ⊃ south, west
    ("reach the uppermost right", 1),          # uppermost ⊃ upper
    ("proceed to the lowermost left", 2),      # lowermost ⊃ lower
    ("go to the topmost left side", 0),        # topmost ⊃ top
    ("head to the bottommost right", 3),       # bottommost ⊃ bottom
    ("reach the centremost cell", 4),          # centremost ⊃ centre
    ("navigate to the northern east", 1),      # northern ⊃ north (+ east) = top-right
]


def heldout_split(seed=0, frac=0.7):
    data = generate_commands()
    rng = np.random.default_rng(seed)
    rng.shuffle(data)
    cut = int(frac * len(data))
    return data[:cut], data[cut:]


def acc(clf, items):
    return float(np.mean([clf.predict(t)[0] == c for t, c in items]))


def main():
    train, test = heldout_split()
    full = generate_commands()

    # обучаем обе модели на одних и тех же командах
    bow = GoalClassifier(BagOfWords([t for t, _ in train])); bow.fit(train, epochs=400)
    cng = GoalClassifier(CharNgram([t for t, _ in train])); cng.fit(train, epochs=400)

    # для теста на новых словоформах словарь признаков строим по ПОЛНОМУ набору команд
    # (но сами новые слова целиком в нём не встречаются — проверяем обобщение по подсловам)
    bow_f = GoalClassifier(BagOfWords([t for t, _ in full])); bow_f.fit(full, epochs=400)
    cng_f = GoalClassifier(CharNgram([t for t, _ in full])); cng_f.fit(full, epochs=400)

    print("▶ Смысл vs ключевые слова: подсловные эмбеддинги против мешка целых слов\n")
    print(f"   признаков: мешок слов = {bow.bow.size};  char-n-граммы = {cng.bow.size}\n")

    print("1) Невиданные КОМБИНАЦИИ знакомых слов (held-out, рекомбинация):")
    print(f"   мешок слов (baseline): {acc(bow, test) * 100:.0f}%     char-n-граммы: {acc(cng, test) * 100:.0f}%")
    print("   (обе модели здесь сильны — это лёгкий случай)\n")

    print(f"2) НОВЫЕ СЛОВОФОРМЫ ({len(NOVEL)} команд, слов целиком НЕ было в обучении) — настоящий тест:")
    print(f"   мешок слов (baseline): {acc(bow_f, NOVEL) * 100:.0f}%     char-n-граммы: {acc(cng_f, NOVEL) * 100:.0f}%\n")

    print("   разбор по командам (✓ = угадал класс по подсловам):")
    print(f"   {'команда':<36}{'ожид.':<12}{'мешок слов':<14}{'char-n-грамм':<14}")
    for text, exp in NOVEL:
        b = bow_f.predict(text)[0]; c = cng_f.predict(text)[0]
        bm = "✓" if b == exp else "✗"; cm = "✓" if c == exp else "✗"
        print(f"   {text:<36}{PLACES[exp][0]:<12}{PLACES[b][0] + ' ' + bm:<14}{PLACES[c][0] + ' ' + cm:<14}")

    print("\n── Что это значит ──")
    print("   Мешок целых слов отбрасывает невиданное слово → на новых словоформах угадывает")
    print("   слабо. Подсловная модель узнаёт слово по его частям и обобщает на формы, которых")
    print("   не видела. Честно: это обобщение по МОРФОЛОГИИ/подсловам (выученное под задачу),")
    print("   а не семантическое понимание; синоним без общих подслов (напр. «middle»→«hub»)")
    print("   так не возьмётся — для этого нужна дистрибутивная семантика из большого корпуса.")


if __name__ == "__main__":
    main()
