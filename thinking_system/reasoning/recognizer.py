"""Распознаватель поиска: по признакам задачи предсказать, какие примитивы пробовать.

Разрыв «поиск экспоненциален, глобальный приор почти не переносится»: вместо одного
униграммного приора на все задачи — модель, ОБУСЛОВЛЕННАЯ задачей. Она читает дешёвые
признаки примеров (тип входа/выхода, отношения чисел/списков) и предсказывает веса
примитивов под ИМЕННО эту задачу. Тогда на невиданных задачах поиск пробует уместные
операции первыми и проверяет на порядок меньше программ (рекогнайзер DreamCoder).

Обучение — гребневая регрессия признаки→использование примитивов (замкнутая форма).
Честно: признаки заданы для домена чисел/списков; это распознаватель под домен, шаг к
масштабируемому поиску, а не универсальный нейро-рекогнайзер.
"""

from __future__ import annotations

import numpy as np

from thinking_system.reasoning.induction import Library, Primitive, default_primitives


def task_features(examples: list[tuple]) -> np.ndarray:
    """Дешёвые признаки задачи (вход→выход) — без решения, до поиска."""
    ins = [i for i, _ in examples]
    outs = [o for _, o in examples]
    il = isinstance(ins[0], list)
    ol = isinstance(outs[0], list)
    f = [1.0, float(il), float(ol)]

    def safe(xs):
        return float(np.mean(xs)) if xs else 0.0

    if not il and not ol:                                   # int → int
        f += [safe([o - i for i, o in examples]),
              safe([o / i for i, o in examples if i]),
              safe([o - i * i for i, o in examples]),
              safe([float(abs(o) > abs(i)) for i, o in examples]),
              safe([float(o == i * i) for i, o in examples])]
    else:
        f += [0.0, 0.0, 0.0, 0.0, 0.0]

    if il:                                                  # список → ?
        li = [len(i) for i in ins]
        lo = [len(o) if isinstance(o, list) else 1 for o in outs]
        si = [sum(i) for i in ins]
        so = [sum(o) if isinstance(o, list) else o for o in outs]
        f += [safe([a / b for a, b in zip(lo, li) if b]),
              safe([a / b for a, b in zip(so, si) if b]),
              safe([float(isinstance(o, list) and o == sorted(o)) for o in outs]),
              safe([float(isinstance(o, list) and o == i[::-1]) for i, o in examples]),
              safe([float(not isinstance(o, list)) for o in outs])]   # список→число (агрегатор)
    else:
        f += [0.0, 0.0, 0.0, 0.0, 0.0]
    return np.array(f, dtype=np.float64)


class Recognizer:
    """Обусловленный задачей приор над примитивами (признаки задачи → веса)."""

    def __init__(self, primitives: list[Primitive] | None = None) -> None:
        self.prims = primitives if primitives is not None else default_primitives()
        self.names = [p.name for p in self.prims]
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.W = None

    def fit(self, tasks: list[list[tuple]], *, max_depth: int = 3, lam: float = 1.0) -> "Recognizer":
        """Обучить признаки задачи → вектор использования примитивов (по найденным решениям)."""
        lib = Library(self.prims)
        X, Y = [], []
        for ex in tasks:
            prog = lib.induce(ex, max_depth=max_depth)
            if prog is None:
                continue
            y = np.zeros(len(self.prims))
            for s in prog.steps:
                if s.name in self.idx:
                    y[self.idx[s.name]] = 1.0
            X.append(task_features(ex)); Y.append(y)
        X, Y = np.array(X), np.array(Y)
        d = X.shape[1]
        self.W = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ Y)
        return self

    def weights(self, examples: list[tuple]) -> dict[str, float]:
        """Предсказать веса примитивов под задачу (для best_first_induce)."""
        pred = task_features(examples) @ self.W
        pred = np.maximum(pred, 0.0) + 0.05                  # положительные веса + база
        return {n: float(pred[i]) for i, n in enumerate(self.names)}
