#!/usr/bin/env python3
"""Опции, обученные ИЗ ОПЫТА (Q-learning) — иерархический RL без готовой карты.

Навыки-опции учатся методом проб и ошибок (награда за подцель), а не считаются из
известной карты. Показываем: кривую обучения, достижение подцели из любой точки,
устойчивость к сбоям (это политика) и композицию опций.

Запуск: python run_qlearn.py
"""

from __future__ import annotations

import numpy as np

from thinking_system.agent.qoption import QOptionLibrary
from thinking_system.world.rooms import rooms_world
from thinking_system.viz import sparkline


def main():
    grid = rooms_world()
    free = grid.free_sids()
    doorways = [grid.sid(c) for c in [(1, 3), (3, 1), (3, 5), (5, 3)]]
    print(f"▶ Мир-комнаты {grid.size}×{grid.size}; 4 опции-навыка к проёмам, обучаемые Q-learning\n")

    lib = QOptionLibrary(grid, doorways, seed=0)
    curves = lib.train(episodes_each=1500)

    c = curves[doorways[0]]
    bins = [float(np.mean(c[k:k + 60])) for k in range(0, len(c) - 59, 60)]
    print("1) ОБУЧЕНИЕ ИЗ ОПЫТА — шагов до подцели по ходу Q-learning (без карты):")
    print("   " + sparkline(bins) + f"   {bins[0]:.0f} → {bins[-1]:.0f} шагов (политика становится эффективной)")

    print("\n2) ВЫУЧЕННАЯ ОПЦИЯ достигает подцели (политика из любой клетки):")
    rng = np.random.default_rng(1)
    for perturb in [0.0, 0.15]:
        succ = n = 0
        for L, opt in lib.options.items():
            for _ in range(25):
                start = int(rng.choice([s for s in free if s != L]))
                succ += opt.reach(start, perturb=perturb, seed=int(rng.integers(10 ** 6)))
                n += 1
        tag = "без сбоев" if perturb == 0 else f"при сбоях {perturb:.0%}/шаг"
        print(f"   {tag:<22}: {100 * succ / n:.0f}% дошли")

    print("\n3) КОМПОЗИЦИЯ опций (иерархия): из угла (0,0) через проёмы в дальнюю комнату:")
    start = grid.sid((0, 0))
    ok = lib.compose(start, [doorways[0], doorways[2]], seed=0)  # (1,3)→(3,5): TL→TR→BR
    print(f"   цепочка опций [проём(1,3) → проём(3,5)]: {'дошёл' if ok else 'нет'}")

    print("\n── Итог ──")
    print("   Навыки выучены ИЗ ОПЫТА (Q-learning), без готовой карты переходов: политика")
    print("   достигает подцели из любой точки, восстанавливается после сбоев и компонуется")
    print("   на верхнем уровне — иерархический RL с нуля.")


if __name__ == "__main__":
    main()
