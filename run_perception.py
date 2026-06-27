#!/usr/bin/env python3
"""Выход за барьер замыкания: перцептивные примитивы, ломающие инварианты.

Геометрический seed (симметрии + перекраска) НЕ может выразить операции, которые
смотрят на содержимое/соседей и агрегируют — это барьер замыкания, а не нехватка
поиска. Добавляем общие перцептивные кирпичи (гравитация, связные объекты, счёт…),
и ТА ЖЕ индукция находит и компонует решения задач, недостижимых раньше.

Запуск: python run_perception.py
"""

from __future__ import annotations

from thinking_system.reasoning.grids import grid_primitives, to_grid, flip_h
from thinking_system.reasoning.perception import perception_primitives, gravity, keep_largest, count_nonzero
from thinking_system.reasoning.induction import induce

GEOM = grid_primitives()
FULL = GEOM + perception_primitives()


def main():
    print("▶ Перцептивные примитивы: выход за барьер замыкания геометрии\n")
    g1 = to_grid([[5, 0], [0, 0], [0, 7]])
    g2 = to_grid([[0, 1, 0], [2, 0, 3], [0, 0, 0]])
    k1 = to_grid([[1, 1, 0, 0], [1, 0, 0, 2], [0, 0, 0, 0]])
    k2 = to_grid([[0, 3, 3, 3], [0, 0, 0, 0], [5, 0, 0, 0]])
    c1 = to_grid([[1, 0, 1], [0, 1, 0]])
    c2 = to_grid([[2, 2], [0, 2]])

    tasks = [
        ("гравитация (уронить вниз)",   [(g, gravity(g)) for g in (g1, g2)]),
        ("крупнейший связный объект",   [(g, keep_largest(g)) for g in (k1, k2)]),
        ("счёт непустых клеток",        [(g, count_nonzero(g)) for g in (c1, c2)]),
        ("гравитация ▸ отражение",      [(g, flip_h(gravity(g))) for g in (g1, g2)]),
    ]

    print("ЗАДАЧА                        | геометрический seed | + перцептивные примитивы")
    for name, tk in tasks:
        pg = induce(tk, GEOM, max_depth=3)
        pf = induce(tk, FULL, max_depth=3)
        print(f"  {name:<28}| {str(pg):<19} | «{pf}»")

    print("\n── Что это значит ──")
    print("   Геометрический seed эти задачи НЕ берёт (None) — это барьер замыкания: из")
    print("   «перетасовать + перекрасить» не вырастает «смотреть на объекты и считать».")
    print("   Добавили общие перцептивные кирпичи (содержимое/соседи/агрегация) — и ТА ЖЕ")
    print("   индукция находит решения и КОМПОНУЕТ их с геометрией («flip_h ▸ gravity»).")
    print("   Это и есть способ дать системе решать такие задачи: не хаки под бенчмарк, а")
    print("   общие примитивы, ломающие инварианты, плюс прежний поиск и рост библиотеки.")


if __name__ == "__main__":
    main()
