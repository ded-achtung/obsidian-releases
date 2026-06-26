"""Грунтинг языка в ДВЕРНЫЕ ПРОЁМЫ мира-комнат: команда → проём (подцель навыка).

Команда на естественном языке («пройди через верхний проём», «иди к восточной
двери») сопоставляется одному из 4 проёмов. Классификатор учится на одних
формулировках, проверяется на ДРУГИХ — если обобщает на невиданные фразы, значит
выучил смысл слов направления, а не запомнил предложения. Понятый проём → выбор
соответствующей ВЫУЧЕННОЙ опции-навыка (см. agent/language_options.py).

Проёмы мира-комнат (rooms_world): вертикальная стена col 3 даёт верхний/нижний
проёмы (1,3)/(5,3); горизонтальная стена row 3 — левый/правый (3,1)/(3,5).
"""

from __future__ import annotations

import numpy as np

from thinking_system.language.grounding import BagOfWords, GoalClassifier

# класс → (клетка-проём, имя, слова-направления)
DOORWAYS: dict[int, tuple[tuple[int, int], str, list[str]]] = {
    0: ((1, 3), "top",    ["top", "upper", "topmost", "north", "northern", "uppermost"]),
    1: ((5, 3), "bottom", ["bottom", "lower", "lowermost", "south", "southern", "bottommost"]),
    2: ((3, 1), "left",   ["left", "leftmost", "west", "western", "left-hand", "leftward"]),
    3: ((3, 5), "right",  ["right", "rightmost", "east", "eastern", "right-hand", "rightward"]),
}
NOUNS = ["doorway", "door", "passage", "opening", "gap"]
VERBS = ["go to", "head to", "move to", "go through", "pass through", "reach", "navigate to", "take", "walk to"]

N_DOORWAYS = len(DOORWAYS)


def doorway_cell(idx: int) -> tuple[int, int]:
    return DOORWAYS[idx][0]


def generate_doorway_commands() -> list[tuple[str, int]]:
    """Все команды (глагол × слово-направления × существительное) с метками проёмов."""
    data: list[tuple[str, int]] = []
    for idx, (_, _, dirs) in DOORWAYS.items():
        for v in VERBS:
            for d in dirs:
                for n in NOUNS:
                    data.append((f"{v} the {d} {n}", idx))
    return data


def train_doorway_classifier(*, test_frac: float = 0.3, epochs: int = 300, seed: int = 0) -> tuple[GoalClassifier, list, list]:
    """Сгенерировать команды, разбить на train/held-out, обучить классификатор.

    Возвращает (классификатор, train, held-out). Точность на held-out показывает
    обобщение на НЕВИДАННЫЕ формулировки (тот же словарь, новые предложения).
    """
    data = generate_doorway_commands()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data))
    cut = int((1 - test_frac) * len(data))
    train = [data[i] for i in idx[:cut]]
    held = [data[i] for i in idx[cut:]]
    clf = GoalClassifier(BagOfWords([t for t, _ in train]), n_classes=N_DOORWAYS)
    clf.fit(train, epochs=epochs)
    return clf, train, held


def accuracy(clf: GoalClassifier, data: list[tuple[str, int]]) -> float:
    return float(np.mean([clf.predict(t)[0] == y for t, y in data]))
