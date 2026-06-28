"""Самообучающийся цикл: провалы ПОРОЖДАЮТ новые схемы, которые система оставляет себе.

Замыкает рефлексию в обучение. Раньше система ставила диагноз «нет схемы» — и всё.
Здесь:

  решить сидовым репертуаром → собрать ПРОВАЛЫ → сгенерировать кандидатов-схемы
  (префикс из атомов ▸ хвост с параметрами из данных) → ОСТАВИТЬ те, что объясняют
  ≥2 задачи (переиспользуемые — критерий сжатия, как wake/sleep DreamCoder) →
  нарастить репертуар → решить больше.

Схемы (`crop ▸ *`, `largest ▸ *`…) система ОТКРЫВАЕТ САМА из своих провалов и КОПИТ —
репертуар больше не фиксирован. Честно: пространство кандидатов (атомы) наше; но какие
схемы полезны и стоит ли их держать — система решает из опыта, а не мы вписываем.
"""

from __future__ import annotations

from collections import Counter

from thinking_system.reasoning.synthesize import ATOMS, _apply_seq, _compose
from thinking_system.reasoning.invent import invent


class SelfImprover:
    """Наращивает репертуар схем из собственных провалов (wake/sleep по схемам)."""

    def __init__(self, *, max_prefix: int = 1) -> None:
        self.schemas: list[list] = []           # выученные префиксы-схемы (списки атомов)
        self.max_prefix = max_prefix
        self.discovery: dict = {}

    def _solve_prefix(self, prefix, train):
        """Решить задачу схемой «префикс ▸ хвост-из-данных»; (имя, fn) или None."""
        mids = [_apply_seq(prefix, i) for i, _ in train]
        if any(m is None for m in mids):
            return None
        name, tail = invent(list(zip(mids, [o for _, o in train])))
        if tail is None:
            return None
        fn = _compose(prefix, tail)
        try:
            if all(fn(i) == o for i, o in train):
                return name, fn
        except Exception:  # noqa: BLE001
            return None
        return None

    # ── базовый репертуар (плоский invent) ───────────────────────────────────────
    @staticmethod
    def _flat_solves(train):
        name, fn = invent(train)
        return (name, fn) if fn is not None else None

    # ── SLEEP: открыть новые схемы из ПРОВАЛОВ ────────────────────────────────────
    def discover(self, failed_tasks, *, min_reuse: int = 2):
        """Из провальных задач найти префиксы-схемы, объясняющие ≥min_reuse задач (на train)."""
        counts: Counter = Counter()
        single = [[a] for a in ATOMS]                        # кандидаты-префиксы длины 1
        for prefix in single:
            label = prefix[0][0]
            for train, _ in failed_tasks:
                if self._solve_prefix(prefix, train) is not None:
                    counts[label] += 1
        kept = [lbl for lbl, c in counts.items() if c >= min_reuse]
        self.schemas = [[a for a in ATOMS if a[0] == lbl][0:1] for lbl in kept]  # сохранить как префиксы
        self.discovery = {"candidate_counts": dict(counts), "kept": kept, "min_reuse": min_reuse}
        return self.discovery

    # ── решение наросшим репертуаром ─────────────────────────────────────────────
    def solve(self, train):
        """Сначала базовый invent; если нет — ВЫУЧЕННЫЕ схемы (наросший репертуар)."""
        flat = self._flat_solves(train)
        if flat is not None:
            return ("flat:" + flat[0], flat[1])
        for prefix in self.schemas:
            r = self._solve_prefix(prefix, train)
            if r is not None:
                return (prefix[0][0] + " ▸ " + r[0], r[1])
        return None, None
