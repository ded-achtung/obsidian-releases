#!/usr/bin/env python3
"""Третий слой примитивов: симметрия под заслонкой, гравитация по сторонам, рамка.

Главная новинка — restore_symmetry: система САМА находит цвет-окклюдер (тот, чьё
стирание оставляет симметричный узор) и достраивает скрытое по зеркалам. Плюс
направленная гравитация, калейдоскоп и обрезка рамки. Все — общие, и та же
индукция их находит и компонует с прежними слоями.

Запуск: python run_objects.py
"""

from __future__ import annotations

from thinking_system.reasoning.grids import grid_primitives, to_grid
from thinking_system.reasoning.perception import perception_primitives
from thinking_system.reasoning.structural import structural_primitives
from thinking_system.reasoning.objects import (object_primitives, restore_symmetry, mirror_quad,
                                               gravity_left, trim_border)
from thinking_system.reasoning.induction import induce

GEOM = grid_primitives()
FULL = GEOM + perception_primitives() + structural_primitives() + object_primitives()


def main():
    print("▶ Третий слой: симметрия под заслонкой, гравитация по сторонам, рамка\n")
    o1 = to_grid([[8, 8, 2, 1], [8, 8, 3, 2], [2, 3, 3, 2], [1, 2, 2, 1]])
    o2 = to_grid([[1, 9, 9, 1], [4, 9, 9, 4], [4, 5, 5, 4], [1, 6, 6, 1]])
    g1 = to_grid([[0, 5, 0, 7], [0, 0, 0, 0]]); g2 = to_grid([[0, 0, 1], [2, 0, 3]])
    q1 = to_grid([[1, 2], [3, 4]]); q2 = to_grid([[5, 0], [0, 6]])
    b1 = to_grid([[5, 5, 5, 5], [5, 1, 2, 5], [5, 3, 4, 5], [5, 5, 5, 5]])
    b2 = to_grid([[7, 7, 7], [7, 9, 7], [7, 7, 7]])
    b3 = to_grid([[6, 6, 6, 6], [6, 7, 8, 6], [6, 9, 1, 6], [6, 6, 6, 6]])

    tasks = [
        ("восстановить симметрию",      [(g, restore_symmetry(g)) for g in (o1, o2)]),
        ("гравитация влево",            [(g, gravity_left(g)) for g in (g1, g2)]),
        ("калейдоскоп 2×2",             [(g, mirror_quad(g)) for g in (q1, q2)]),
        ("снять рамку",                 [(g, trim_border(g)) for g in (b1, b2)]),
        ("снять рамку ▸ калейдоскоп",    [(g, mirror_quad(trim_border(g))) for g in (b1, b3)]),
    ]

    print("ЗАДАЧА                        | геометрический seed | + все слои восприятия")
    for name, tk in tasks:
        pg = induce(tk, GEOM, max_depth=3)
        pf = induce(tk, FULL, max_depth=3)
        print(f"  {name:<28}| {str(pg):<19} | «{pf}»")

    print("\n── Что это значит ──")
    print("   restore_symmetry сам находит цвет-заслонку и чинит узор — не зная цвета заранее.")
    print("   Словарь восприятия растёт слоями (геометрия → перцепция → структура → объекты),")
    print("   и ТА ЖЕ индукция всё это компонует. Это путь к новым семействам задач без")
    print("   подгонки под бенчмарк: больше общих кирпичей + прежний поиск + рост библиотеки.")


if __name__ == "__main__":
    main()
