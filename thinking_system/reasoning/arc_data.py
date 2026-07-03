"""Загрузка реального датасета ARC-AGI (Chollet) — замер воспроизводим ИЗ репозитория.

Раньше «замер на ARC» существовал только как утверждение в docstring/коммитах:
данных в репозитории не было, и проверить числа было нельзя. Этот модуль убирает
разрыв: задачи ARC загружаются из проверяемых источников, по убыванию приоритета:

  1) локальный каталог в каноническом формате ARC:
     <data_dir>/training/*.json и <data_dir>/evaluation/*.json
     (по умолчанию data/arc/ в корне проекта);
  2) пакет `arckit` (pip install arckit) — датасет ARC-AGI-1 идёт в комплекте
     (400 training + 400 evaluation).

Каждая задача: train-пары (демонстрации) и test-пары (скрытые входы для проверки).
Сетки конвертируются в Grid (кортеж кортежей int) из reasoning.grids.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from thinking_system.reasoning.grids import Grid, to_grid

Pair = tuple  # (вход: Grid, выход: Grid)


@dataclass(frozen=True)
class ArcTask:
    """Одна задача ARC: id + демонстрационные пары + скрытые тест-пары."""

    task_id: str
    train: tuple  # ((Grid, Grid), ...) — по ним ищется программа
    test: tuple   # ((Grid, Grid), ...) — по ним проверяется (exact match)


def _pairs(items) -> tuple:
    return tuple((_grid(p["input"]), _grid(p["output"])) for p in items)


def _grid(rows) -> Grid:
    return to_grid([[int(v) for v in r] for r in rows])


def load_arc_dir(path: str) -> list[ArcTask]:
    """Каталог *.json в каноническом формате ARC → список задач (сортировка по id)."""
    tasks: list[ArcTask] = []
    for fname in sorted(os.listdir(path)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(path, fname), encoding="utf-8") as f:
            raw = json.load(f)
        tasks.append(ArcTask(fname[:-5], _pairs(raw["train"]), _pairs(raw["test"])))
    return tasks


def _load_arckit(split: str) -> list[ArcTask]:
    import arckit  # noqa: PLC0415 — опциональная зависимость

    train_set, eval_set = arckit.load_data("arcagi")           # ARC-AGI-1: 400 + 400
    task_set = train_set if split == "training" else eval_set

    def np_pairs(pairs) -> tuple:
        return tuple((_grid(i.tolist()), _grid(o.tolist())) for i, o in pairs)

    return [ArcTask(t.id, np_pairs(t.train), np_pairs(t.test)) for t in task_set]


def load_arc(split: str = "training", data_dir: str | None = None) -> list[ArcTask]:
    """Загрузить сплит ARC-AGI-1 ('training' | 'evaluation').

    Сначала локальный каталог (data_dir или ./data/arc/<split>), затем arckit.
    Бросает RuntimeError с инструкцией, если данных нет ни там, ни там.
    """
    if split not in ("training", "evaluation"):
        raise ValueError(f"неизвестный сплит: {split!r}")
    base = data_dir if data_dir is not None else os.path.join("data", "arc")
    local = os.path.join(base, split)
    if os.path.isdir(local):
        return load_arc_dir(local)
    try:
        return _load_arckit(split)
    except ImportError:
        raise RuntimeError(
            "Данные ARC не найдены. Либо `pip install arckit` (датасет в комплекте), "
            f"либо положите JSON-задачи в {local}/ (канонический формат ARC)."
        ) from None
