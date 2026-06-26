"""Последовательность РАЗЛИЧНЫХ задач (режимов) для проверки забывания.

Каждая задача — синтетический поток со своей динамикой (своя частотная полоса,
фазы и проекция). Задачи предъявляются по очереди; одна общая модель мира должна
их все удерживать. Без повторения (replay) обучение новой задаче затирает старые
(катастрофическое забывание) — это и проверяет шаг 3.

Задача — это ФАБРИКА потока: и обучение, и оценка берут свежий поток с той же
динамикой (детерминированный по сиду), что даёт фиксированный eval-набор.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from thinking_system.streams.base import Stream
from thinking_system.streams.synthetic import SyntheticStream


@dataclass(frozen=True)
class Task:
    """Именованная задача с фабрикой потока.

    Attributes:
        name: имя задачи (например, "task1").
        _factory: callable, возвращающий свежий Stream с динамикой задачи.
    """

    name: str
    _factory: Callable[[], Stream]

    def stream(self) -> Stream:
        """Свежий поток данных этой задачи."""
        return self._factory()


def make_task_sequence(
    n_tasks: int = 4,
    *,
    obs_dim: int = 32,
    n_modes: int = 2,
    band_width: float = 0.4,
    band_gap: float = 0.6,
    noise: float = 0.02,
    seed: int = 0,
) -> list[Task]:
    """Построить последовательность задач с НЕПЕРЕСЕКАЮЩИМИСЯ частотными полосами.

    Задача i занимает полосу [lo_i, lo_i+band_width], lo_i = 0.4 + i*band_gap —
    динамики гарантированно различны, поэтому забывание хорошо заметно.
    """
    tasks: list[Task] = []
    for i in range(n_tasks):
        lo = 0.4 + i * band_gap
        hi = lo + band_width
        task_seed = seed + 100 * (i + 1)

        def factory(s: int = task_seed, flo: float = lo, fhi: float = hi) -> Stream:
            return SyntheticStream(
                obs_dim=obs_dim, n_modes=n_modes, noise=noise, freq_lo=flo, freq_hi=fhi, seed=s
            )

        tasks.append(Task(name=f"task{i + 1}", _factory=factory))
    return tasks
