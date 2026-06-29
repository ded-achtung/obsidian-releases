"""Тест самообучения: провалы порождают переиспользуемую схему, репертуар растёт."""

from __future__ import annotations

from thinking_system.agent.self_improve import SelfImprover
from thinking_system.reasoning.invent import invent


def _crop_recolor(g):
    cells = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v != 0]
    r0 = min(r for r, _ in cells); r1 = max(r for r, _ in cells)
    c0 = min(c for _, c in cells); c1 = max(c for _, c in cells)
    return tuple(tuple(g[r][c] + 4 for c in range(c0, c1 + 1)) for r in range(r0, r1 + 1))


# три задачи, каждой нужна схема «crop ▸ colormap» (плоский invent их не берёт)
GA = ((0, 0, 0, 0), (0, 1, 2, 0), (0, 3, 4, 0), (0, 0, 0, 0))
GB = ((0, 0, 0), (0, 2, 1), (0, 4, 3))
GC = ((0, 0, 0, 0, 0), (0, 1, 3, 0, 0), (0, 2, 4, 0, 0), (0, 0, 0, 0, 0))


def _task(g):
    return [(g, _crop_recolor(g))], [(g, _crop_recolor(g))]


def test_flat_invent_cannot_solve_these() -> None:
    """Эти задачи меняют форму → плоский invent (без префикса) их не решает."""
    for g in (GA, GB, GC):
        train, _ = _task(g)
        name, fn = invent(train)
        assert fn is None                              # нужна композиция, которой во flat нет


def test_discovers_reusable_schema_from_failures() -> None:
    """Из провалов система находит, что префикс crop объясняет ≥2 задачи, и оставляет его."""
    imp = SelfImprover()
    failed = [_task(GA), _task(GB), _task(GC)]
    disc = imp.discover(failed, min_reuse=2)
    assert disc["candidate_counts"].get("crop", 0) >= 2
    assert "crop" in disc["kept"]
    assert any(s[0][0] == "crop" for s in imp.schemas)  # схема добавлена в репертуар


def test_grown_repertoire_solves_what_seed_could_not() -> None:
    """Наросший репертуар решает задачу, которую сидовый (плоский) не брал."""
    imp = SelfImprover()
    imp.discover([_task(GA), _task(GB)], min_reuse=2)   # выучить crop из двух
    # сид не решает GC; наросший — решает
    assert imp._flat_solves(_task(GC)[0]) is None
    label, fn = imp.solve(_task(GC)[0])
    assert fn is not None and "crop" in label
    g = ((0, 0, 0, 0), (0, 4, 2, 0), (0, 1, 3, 0), (0, 0, 0, 0))
    assert fn(g) == _crop_recolor(g)                    # и обобщает (цвета виденные)
