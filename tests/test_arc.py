"""Тесты ARC-харнесса: загрузка формата, честная оценка на held-out, без зависимости от данных.

Синтетические задачи в формате ARC проверяют харнесс самодостаточно. Реальный набор
(data/arc) — отдельным тестом, который ПРОПУСКАЕТСЯ, если данных нет (в репо их нет).
"""

from __future__ import annotations

import json
import os

import pytest

from thinking_system.arc import evaluate_arc, load_arc_tasks, solve_task


def _write_task(d, tid, train, test):
    with open(os.path.join(d, f"{tid}.json"), "w", encoding="utf-8") as f:
        json.dump({"train": [{"input": i, "output": o} for i, o in train],
                   "test": [{"input": i, "output": o} for i, o in test]}, f)


def test_loads_arc_format(tmp_path) -> None:
    _write_task(tmp_path, "t1", [([[1, 2]], [[2, 1]])], [([[3, 4]], [[4, 3]])])
    tasks = load_arc_tasks(str(tmp_path))
    assert len(tasks) == 1 and tasks[0]["id"] == "t1"
    assert tasks[0]["train"][0]["input"] == [[1, 2]]


def test_solves_flip_task_on_heldout() -> None:
    # Сетка-флип: правило из train, проверка на ОТЛОЖЕННОМ test (точное совпадение).
    task = {"id": "flip", "test": [{"input": [[9, 0], [1, 2]], "output": [[0, 9], [2, 1]]}],
            "train": [{"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
                      {"input": [[5, 6], [7, 8]], "output": [[6, 5], [8, 7]]}]}
    prog, ok = solve_task(task)
    assert ok and prog is not None


def test_solves_recolor_task_on_heldout() -> None:
    task = {"id": "recolor", "test": [{"input": [[1, 1, 2, 2]], "output": [[2, 2, 1, 1]]}],
            "train": [{"input": [[1, 2, 1]], "output": [[2, 1, 2]]},
                      {"input": [[2, 2, 1]], "output": [[1, 1, 2]]}]}
    prog, ok = solve_task(task)
    assert ok and prog is not None


def test_out_of_scope_task_not_falsely_solved() -> None:
    # Подсчёт (3x3 → 1x1) — вне репертуара: честно НЕ решено.
    task = {"id": "count", "test": [{"input": [[1, 0], [0, 0]], "output": [[1]]}],
            "train": [{"input": [[1, 1], [1, 0]], "output": [[3]]},
                      {"input": [[1, 1], [1, 1]], "output": [[4]]}]}
    _prog, ok = solve_task(task)
    assert not ok


@pytest.mark.skipif(not os.path.isdir("data/arc/training"), reason="ARC data not present (download separately)")
def test_real_arc_harness_runs() -> None:
    res = evaluate_arc("data/arc/training", limit=60)
    assert 0 <= res["solved"] <= res["total"] == 60      # харнесс работает на реальных данных
