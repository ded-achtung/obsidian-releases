"""Загрузка и честный замер на ВНЕШНИХ задачах ARC (данные в репозиторий не входят).

ARC-AGI (François Chollet, лицензия Apache-2.0): каждая задача — `train` (примеры
вход→выход сеток) + `test` (отложенные). Формат JSON:
    {"train":[{"input":grid,"output":grid},…], "test":[{"input":grid,"output":grid},…]}
Сетки — списки списков целых 0–9 (цвета). Грузим из ЛОКАЛЬНОЙ папки `*.json`.

Замер ЧЕСТНЫЙ и БЕЗ УТЕЧКИ: правило выводится из `train` задачи и проверяется на её
`test`-входе — задача засчитана только если ТОЧНО совпали ВСЕ test-выходы (ARC так и
оценивает). Никакого подглядывания в ответ.

Это узкий grid-движок (flip/rot/transpose/перекраска/морфология, композиция ≤2) против
очень разнообразного ARC, поэтому покрытие ожидаемо НИЗКОЕ. Низкая честная цифра here
ценнее завышенной: она показывает реальную ШИРИНУ примитивов, а не подгонку под свой
набор. Данные качаются отдельно (см. README), в репозиторий не коммитятся.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Callable

from thinking_system.system import ThinkingSystem


def load_arc_tasks(path: str) -> list[dict]:
    """Загрузить задачи ARC из папки с `*.json` (формат ARC-AGI)."""
    tasks = []
    for f in sorted(glob.glob(os.path.join(path, "*.json"))):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        tasks.append({"id": os.path.splitext(os.path.basename(f))[0], "train": d["train"], "test": d["test"]})
    return tasks


def _grid(g: Any) -> tuple:
    """Нормализовать сетку к кортежу кортежей целых (для сравнения вход/выход)."""
    return tuple(tuple(int(v) for v in row) for row in g)


def solve_task(task: dict, *, make: Callable[[], Any] = ThinkingSystem) -> tuple[str | None, bool]:
    """Вывести правило из train задачи и проверить на ВСЕХ её test-входах (точно)."""
    train = [(_grid(ex["input"]), _grid(ex["output"])) for ex in task["train"]]
    try:
        prog = make().solve(train)
    except Exception:  # noqa: BLE001 — задача не должна ронять весь прогон
        prog = None
    if prog is None:
        return None, False
    try:
        ok = all(_grid(prog(_grid(ex["input"]))) == _grid(ex["output"]) for ex in task["test"])
    except Exception:  # noqa: BLE001 — правило неприменимо к test-входу
        ok = False
    return str(prog), ok


def evaluate_arc(path: str, *, limit: int | None = None, make: Callable[[], Any] = ThinkingSystem) -> dict:
    """Прогнать набор ARC из папки и вернуть честное покрытие (точное совпадение test)."""
    tasks = load_arc_tasks(path)
    if limit is not None:
        tasks = tasks[:limit]
    solved = []
    for task in tasks:
        prog, ok = solve_task(task, make=make)
        if ok:
            solved.append((task["id"], prog))
    return {"total": len(tasks), "solved": len(solved), "solved_tasks": solved}
