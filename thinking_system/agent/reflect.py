"""Рефлексивный решатель: система НАБЛЮДАЕТ свои попытки, понимает что работает,
анализирует ПОЧЕМУ не сработало.

Раньше решатель просто пробовал гипотезы и возвращал подошедшую — он слеп к себе.
Здесь добавлен метакогнитивный слой:
  • НАБЛЮДЕНИЕ — для каждой задачи пишется trace: что пробовал, что подошло, что решило,
    а если нет — ПОЧЕМУ (нет схемы / отверг как зубрёжку / подошло на train, не обобщилось);
  • ПОНИМАНИЕ ЧТО РАБОТАЕТ — из наблюдений строится модель собственной компетентности,
    предсказывающая «смогу ли я эту задачу» ДО решения (метапознание);
  • АНАЛИЗ ПРОВАЛОВ — агрегированный разбор «почему не вышло», т.е. система понимает
    границу своих умений.

Честно: это рефлексия над СВОИМ перебором схем (не над миром); но это именно «думать о
том, что обнаружил, и почему не сработало», чего раньше не было.
"""

from __future__ import annotations

import numpy as np

from thinking_system.reasoning.invent import (
    INVENTORS, _shape, _factors, _build_table, _apply_table,
)


def grid_features(train) -> np.ndarray:
    """Дешёвые признаки грид-задачи (формы, цвета, отношения) — для метапознания."""
    irs = [_shape(i)[0] for i, _ in train]; ics = [_shape(i)[1] for i, _ in train]
    ors = [_shape(o)[0] for _, o in train]; ocs = [_shape(o)[1] for _, o in train]
    m = lambda x: float(np.mean(x)) if x else 0.0
    same = all(_shape(i) == _shape(o) for i, o in train)
    sub = all(_shape(o)[0] <= _shape(i)[0] and _shape(o)[1] <= _shape(i)[1] for i, o in train)
    in_colors = len({v for i, _ in train for row in i for v in row})
    out_colors = len({v for _, o in train for row in o for v in row})
    return np.array([
        1.0, m(irs), m(ics), m(ors), m(ocs),
        m([orr / ir for orr, ir in zip(ors, irs) if ir]),
        m([oc / ic for oc, ic in zip(ocs, ics) if ic]),
        1.0 if same else 0.0,
        1.0 if _factors(train, up=True) else 0.0,
        1.0 if _factors(train, up=False) else 0.0,
        1.0 if sub else 0.0,
        float(in_colors), float(out_colors), float(out_colors - in_colors),
        float(len(train)),
    ])


def shape_class(train) -> str:
    """Как связаны формы входа и выхода — грубая классификация задачи."""
    if all(_shape(i) == _shape(o) for i, o in train):
        return "same_shape"
    if _factors(train, up=True):
        return "integer_upscale"
    if _factors(train, up=False):
        return "integer_downscale"
    if all(_shape(o)[0] <= _shape(i)[0] and _shape(o)[1] <= _shape(i)[1] for i, o in train):
        return "sub_region"
    return "unrelated"


def _would_memorize(train) -> bool:
    """Нашёл бы локальное правило, подходящее на train, но проваливающее обобщение (LOO)?"""
    if len(train) < 2 or not all(_shape(i) == _shape(o) for i, o in train):
        return False
    plus = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    table = _build_table(train, plus)
    if table is None:
        return False
    fits = all(_apply_table(table, i, plus) == o for i, o in train)
    if not fits:
        return False
    for h in range(len(train)):
        sub = _build_table(train[:h] + train[h + 1:], plus)
        if sub is None or _apply_table(sub, train[h][0], plus) != train[h][1]:
            return True                                    # подошло, но не обобщается → зубрёжка
    return False


class ReflectiveSolver:
    """Решатель, который наблюдает свои попытки, понимает что работает и почему провал."""

    def __init__(self) -> None:
        self.episodes: list[dict] = []
        self._W = None

    # ── НАБЛЮДЕНИЕ ────────────────────────────────────────────────────────────────
    def solve_and_observe(self, train, test, *, task_id=None) -> dict:
        ep = {"id": task_id, "features": grid_features(train), "shape": shape_class(train),
              "fits": [], "solved_by": None, "reason": None}
        for name, synth in INVENTORS:
            try:
                fn = synth(train)
            except Exception:  # noqa: BLE001
                fn = None
            if fn is None:
                continue
            try:
                if not all(fn(i) == o for i, o in train):
                    continue
            except Exception:  # noqa: BLE001
                continue
            ep["fits"].append(name)
            try:
                if test and all(fn(t) == o for t, o in test):
                    ep["solved_by"] = name
                    break
            except Exception:  # noqa: BLE001
                pass
        if ep["solved_by"] is None:
            ep["reason"] = self._diagnose(train, ep)        # ПОЧЕМУ не сработало
        self.episodes.append(ep)
        return ep

    def _diagnose(self, train, ep) -> str:
        if ep["fits"]:
            return "fit_train_failed_test"                  # подошло на train, не обобщилось
        if _would_memorize(train):
            return "rejected_as_memorization"               # мог бы зазубрить — отказался (сам понял)
        return f"no_hypothesis:{ep['shape']}"               # нет схемы под этот класс

    # ── ПОНИМАНИЕ ЧТО РАБОТАЕТ (метапознание) ────────────────────────────────────
    def build_competence(self, *, lam: float = 1.0) -> "ReflectiveSolver":
        """Из наблюдений выучить признаки→решаемо + порог (метапознание; редкий класс)."""
        X = np.array([e["features"] for e in self.episodes])
        y = np.array([1.0 if e["solved_by"] else 0.0 for e in self.episodes])
        d = X.shape[1]
        self._W = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
        # порог калибруем по наблюдениям (максимум F1) — решаемых мало, 0.5 не годится
        s = X @ self._W
        best_thr, best_f1 = 0.5, -1.0
        for thr in np.unique(s):
            pred = s >= thr
            tp = float(np.sum(pred & (y > 0))); fp = float(np.sum(pred & (y == 0)))
            fn = float(np.sum(~pred & (y > 0)))
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if f1 > best_f1:
                best_f1, best_thr = f1, float(thr)
        self._thr = best_thr
        return self

    def predict_solvable(self, train) -> float:
        """Метакогнитивная оценка «смогу ли я это» ДО попытки решить."""
        return float(grid_features(train) @ self._W)

    def thinks_it_can_solve(self, train) -> bool:
        """Метакогнитивное решение «возьмусь ли я» (по выученному порогу)."""
        return self.predict_solvable(train) >= self._thr

    # ── АНАЛИЗ СВОИХ ДЕЙСТВИЙ ─────────────────────────────────────────────────────
    def reflect(self) -> dict:
        """Самоотчёт: что решаю (каким синтезатором) и ПОЧЕМУ проваливаю остальное."""
        from collections import Counter
        solved = [e for e in self.episodes if e["solved_by"]]
        failed = [e for e in self.episodes if not e["solved_by"]]
        return {
            "n": len(self.episodes),
            "solved": len(solved),
            "by_synth": dict(Counter(e["solved_by"] for e in solved)),
            "solved_shapes": dict(Counter(e["shape"] for e in solved)),
            "failure_reasons": dict(Counter(e["reason"].split(":")[0] for e in failed)),
            "failure_shapes": dict(Counter(e["shape"] for e in failed)),
        }
