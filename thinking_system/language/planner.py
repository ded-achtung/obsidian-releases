"""Многошаговые задачи: спланировать ПОСЛЕДОВАТЕЛЬНОСТЬ правил к цели.

Иногда одной операции мало. Дано «получи <цель> из <входа>» — система ИЩЕТ
композицию выученных операций, превращающую вход в цель (поиск в ширину по
значениям = планирование). Никто не называет шаги — система находит план сама и
проверяет его исполнением.

Это перенос идеи планирования (как в мире-сетке) на ДАННЫЕ: состояния — значения,
действия — выученные операции, цель — заданный результат.
"""

from __future__ import annotations

from collections import deque

from thinking_system.language.understanding import extract_arg


def parse_goal_task(task: str) -> tuple:
    """«получи ЦЕЛЬ из ВХОД» / «преврати ВХОД в ЦЕЛЬ» → (вход, цель)."""
    if " из " in task:
        left, right = task.split(" из ", 1)
        return extract_arg(right), extract_arg(left)
    if " в " in task:
        left, right = task.split(" в ", 1)
        return extract_arg(left), extract_arg(right)
    return None, None


def plan_ops(start, goal, words: dict, *, max_depth: int = 4) -> list[str] | None:
    """Поиск в ширину: кратчайшая цепочка операций start→goal (или None)."""
    seen = {repr(start)}
    q = deque([(start, [])])
    while q:
        v, path = q.popleft()
        if v == goal:
            return path
        if len(path) >= max_depth:
            continue
        for name, prog in words.items():
            try:
                nv = prog(v)
            except (TypeError, ValueError, IndexError, KeyError):  # операция неприменима к значению → пропуск
                continue
            if repr(nv) not in seen:
                seen.add(repr(nv))
                q.append((nv, path + [name]))
    return None


class TaskPlanner:
    """Планирует цепочку выученных операций для достижения заданной цели."""

    def __init__(self, lexicon) -> None:
        self.lex = lexicon

    def plan(self, task: str, *, max_depth: int = 4) -> dict:
        start, goal = parse_goal_task(task)
        if start is None or goal is None:
            return {"solved": False, "reason": "не понял вход или цель"}
        path = plan_ops(start, goal, self.lex.words, max_depth=max_depth)
        if path is None:
            return {"solved": False, "reason": "не нашёл план"}
        v = start
        for name in path:
            v = self.lex.words[name](v)
        return {"solved": True, "plan": path, "answer": v}
