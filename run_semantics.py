#!/usr/bin/env python3
"""Дистрибутивная семантика: синоним без общих букв грунтится по смыслу из корпуса.

Разрыв «подслова не берут синонимы без общей формы». Учим эмбеддинги слов из корпуса
(PPMI+SVD). Синонимы ТЕСТА («hub», «core», «portside»…) не были в размеченных командах
и не делят подслов со знакомыми словами — но в корпусе они встречаются в тех же
контекстах. Грунтим команду по близости в пространстве эмбеддингов к знакомым словам.

Бейзлайн — подсловная модель (char-n-граммы): на таких синонимах падает до случайного.

Запуск: python run_semantics.py
"""

from __future__ import annotations

import numpy as np

from thinking_system.language.distributional import DistributionalEmbeddings
from thinking_system.language.grounding import CharNgram, GoalClassifier

# 5 концептов: знакомые слова (есть в размеченных командах) и ТЕСТ-синонимы
# (нет в обучении, не делят подслов со знакомыми — только смысл из контекста).
CONCEPTS = {
    0: dict(name="верх",   known=["top", "upper", "north"],   test=["summit", "ceiling", "overhead"]),
    1: dict(name="низ",    known=["bottom", "lower", "south"], test=["floor", "basement", "underside"]),
    2: dict(name="лево",   known=["left", "west", "leftward"], test=["portside", "larboard"]),
    3: dict(name="право",  known=["right", "east", "rightward"], test=["starboard", "dexter"]),
    4: dict(name="центр",  known=["center", "centre", "middle"], test=["hub", "core", "heart"]),
}
FILLER = ["go", "to", "the", "move", "reach", "area", "side"]


def build_corpus(seed=0, per_concept=400):
    """Корпус: концепт-связные предложения (синонимы концепта со-встречаются)."""
    rng = np.random.default_rng(seed)
    corpus = []
    for cfg in CONCEPTS.values():
        words = cfg["known"] + cfg["test"]
        for _ in range(per_concept):
            k = int(rng.integers(4, 7))
            sent = list(rng.choice(words, k)) + list(rng.choice(FILLER, 2))
            rng.shuffle(sent)
            corpus.append([str(w) for w in sent])
    return corpus


def main():
    print("▶ Дистрибутивная семантика: смысл синонима из контекстов корпуса (PPMI+SVD)\n")
    corpus = build_corpus()
    emb = DistributionalEmbeddings(dim=16).fit(corpus)

    # центроиды концептов из ЗНАКОМЫХ слов (как разметка)
    cents = {cid: emb.phrase(cfg["known"]) for cid, cfg in CONCEPTS.items()}
    cm = np.array([cents[c] for c in sorted(cents)])

    # ТЕСТ: классифицировать команды с ТЕСТ-синонимами (знакомых слов в них нет)
    test_cmds = []
    for cid, cfg in CONCEPTS.items():
        for w in cfg["test"]:
            test_cmds.append((f"go to the {w}", cid))

    # 1) дистрибутивная семантика: ближайший центроид в пространстве эмбеддингов
    def dist_predict(text):
        words = text.split()
        q = emb.phrase(words)
        if q is None:
            return -1
        return int(np.argmin(((cm - q) ** 2).sum(1)))
    dist_acc = np.mean([dist_predict(t) == c for t, c in test_cmds])

    # 2) бейзлайн — подсловная модель, обученная на ЗНАКОМЫХ словах
    train_cmds = []
    for cid, cfg in CONCEPTS.items():
        for w in cfg["known"]:
            for v in ("go to the", "move to the", "reach the"):
                train_cmds.append((f"{v} {w}", cid))
    bow = GoalClassifier(CharNgram([t for t, _ in train_cmds]), n_classes=5)
    bow.fit(train_cmds, epochs=400)
    sub_acc = np.mean([bow.predict(t)[0] == c for t, c in test_cmds])

    print(f"   корпус: {len(corpus)} предложений; эмбеддинги dim={emb.dim}; "
          f"тест: {len(test_cmds)} команд с СИНОНИМАМИ\n")
    print(f"   {'метод':<40}{'точность на синонимах':>22}")
    print(f"   {'подслова (char-n-граммы, бейзлайн)':<40}{sub_acc * 100:>21.0f}%")
    print(f"   {'дистрибутивная семантика (PPMI+SVD)':<40}{dist_acc * 100:>21.0f}%\n")

    print("   разбор (синоним → концепт):")
    for t, c in test_cmds:
        w = t.split()[-1]
        d = dist_predict(t); b = bow.predict(t)[0]
        print(f"   {w:<12} ожид. {CONCEPTS[c]['name']:<7} | подслова {CONCEPTS.get(b, {'name':'?'})['name']:<7} {'✓' if b==c else '✗'}"
              f" | семантика {CONCEPTS[d]['name'] if d>=0 else '?':<7} {'✓' if d==c else '✗'}")

    print("\n── Что это значит ──")
    print("   «hub», «core», «portside» не делят букв со знакомыми словами, и подсловная модель")
    print("   их не берёт. Но в корпусе они стоят в тех же контекстах, что «center», «left» — и")
    print("   дистрибутивные эмбеддинги ставят их рядом, так что грунтинг работает по СМЫСЛУ, а")
    print("   не по форме. Честно: смысл здесь = близость по употреблению в корпусе (синоним")
    print("   должен встречаться в корпусе); это не понимание, но шаг от формы к значению.")


if __name__ == "__main__":
    main()
