#!/usr/bin/env python3
"""Второй слой выхода за барьер: структурно-объектные примитивы.

После перцептивных кирпичей (gravity/объекты/счёт) добавляем структурный слой:
достройка симметрии, фрактал, контур, счёт объектов, модальный цвет. Каждый —
общая операция, не выразимая композицией геометрии; ТА ЖЕ индукция их находит и
компонует. Это покрывает новые семейства ARC (симметрия, фрактал, граница).

Запуск: python run_structural.py
"""

from __future__ import annotations

from thinking_system.reasoning.grids import grid_primitives, to_grid, flip_h
from thinking_system.reasoning.perception import perception_primitives
from thinking_system.reasoning.structural import (structural_primitives, complete_symmetry,
                                                  replicate_by_self, outline)
from thinking_system.reasoning.induction import induce

GEOM = grid_primitives()
FULL = GEOM + perception_primitives() + structural_primitives()


def main():
    print("▶ Структурно-объектные примитивы: второй слой за барьером замыкания\n")
    f1 = to_grid([[1, 0], [0, 1]]); f2 = to_grid([[0, 2], [2, 2]])
    h1 = to_grid([[0, 2, 2, 1], [2, 3, 3, 2], [2, 3, 3, 2], [1, 2, 2, 1]])
    h2 = to_grid([[5, 0, 0, 5], [0, 7, 7, 0], [0, 7, 7, 0], [5, 6, 0, 5]])
    o1 = to_grid([[1, 1, 1], [1, 1, 1], [1, 1, 1]]); o2 = to_grid([[5, 5], [5, 5]])

    tasks = [
        ("достроить симметрию",        [(g, complete_symmetry(g)) for g in (h1, h2)]),
        ("фрактал (вход на себя)",      [(g, replicate_by_self(g)) for g in (f1, f2)]),
        ("контур объектов",             [(g, outline(g)) for g in (o1, o2)]),
        ("фрактал ▸ отражение",         [(g, flip_h(replicate_by_self(g))) for g in (f1, f2)]),
    ]

    print("ЗАДАЧА                        | геометрический seed | + перцепция + структура")
    for name, tk in tasks:
        pg = induce(tk, GEOM, max_depth=3)
        pf = induce(tk, FULL, max_depth=3)
        print(f"  {name:<28}| {str(pg):<19} | «{pf}»")

    print("\n── Что это значит ──")
    print("   Каждый новый примитив общий (симметрия/фрактал/граница/счёт), а не решатель")
    print("   под задачу. Геометрия их не выражает (барьер), но ТА ЖЕ индукция находит и")
    print("   КОМПОНУЕТ их с геометрией («fractal ▸ flip_h»). Путь к новым семействам ARC —")
    print("   ещё общие перцептивные/структурные кирпичи, а не подгонка под бенчмарк.")


if __name__ == "__main__":
    main()
