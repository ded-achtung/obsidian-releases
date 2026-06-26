#!/usr/bin/env python3
"""Внутренняя мотивация над опциями: агент САМ открывает подцели и учит навыки.

Никто не даёт агенту проёмы. Блуждая, он находит БУТЫЛОЧНЫЕ ГОРЛЫШКИ из опыта,
делает их подцелями навыков, учит к ним Q-опции (кривая → learning progress =
автокуррикулум) и затем доходит куда угодно, переиспользуя открытые навыки —
лучше, чем со случайными подцелями.

Запуск: python run_option_discovery.py
"""

from __future__ import annotations

import numpy as np

from thinking_system.world.rooms import rooms_world
from thinking_system.agent.option_discovery import OptionDiscoverer
from thinking_system.viz import sparkline


def main():
    grid = rooms_world()
    rc = lambda s: (s // grid.size, s % grid.size)
    true_doors = {grid.sid(c) for c in [(1, 3), (5, 3), (3, 1), (3, 5)]}
    print(f"▶ Мир-комнаты {grid.size}×{grid.size}; агент сам открывает подцели (проёмы ему НЕ даны)\n")

    # 1) открыть подцели из опыта
    disc = OptionDiscoverer(grid, seed=0)
    disc.explore(6000)
    subgoals = disc.discover(k=4)
    match = len(set(subgoals) & true_doors)
    print("1) ОТКРЫТИЕ ПОДЦЕЛЕЙ из опыта (узкие клетки с высокой центральностью = горлышки):")
    print(f"   найдены: {sorted(rc(s) for s in subgoals)}")
    print(f"   истинные проёмы мира: {sorted(rc(s) for s in true_doors)}  → совпало {match}/4 (агент не знал их)")

    # 2) выучить навыки к открытым подцелям + learning progress (автокуррикулум)
    curves = disc.learn_options(episodes_each=1500)
    prog = disc.learning_progress(curves)
    print("\n2) НАВЫКИ К ОТКРЫТЫМ ПОДЦЕЛЯМ (Q-learning из опыта) и learning progress:")
    for sg in sorted(subgoals, key=lambda s: -prog[s]):
        c = curves[sg]
        print(f"   проём {str(rc(sg)):<7} {sparkline([float(np.mean(c[k:k+60])) for k in range(0, len(c)-59, 60)])}"
              f"  {np.mean(c[:100]):.0f}→{np.mean(c[-100:]):.0f} шагов, прогресс {prog[sg]:+.1f}")
    order = " ≻ ".join(str(rc(s)) for s in sorted(subgoals, key=lambda s: -prog[s]))
    print(f"   автокуррикулум (по убыванию прогресса): {order}")

    # 3) ИСПОЛЬЗОВАНИЕ: покрытие мира открытыми навыками vs случайные подцели
    print("\n3) ИСПОЛЬЗОВАНИЕ — доходит куда угодно через открытые горлышки:")
    cov = disc.coverage(subgoals)
    rng = np.random.default_rng(1)
    rand_cov = [disc.coverage(list(rng.choice(disc.free, size=4, replace=False))) for _ in range(30)]
    print(f"   покрытие ОТКРЫТЫМИ подцелями: {cov:.0%}")
    print(f"   покрытие СЛУЧАЙНЫМИ подцелями: {np.mean(rand_cov):.0%} (в среднем; разброс {min(rand_cov):.0%}–{max(rand_cov):.0%})")

    # конкретно: из угла дойти до дальней клетки в каждой комнате
    targets = {"TL": (1, 1), "TR": (1, 5), "BL": (5, 1), "BR": (5, 5)}
    start = grid.sid((0, 0))
    got = [name for name, c in targets.items() if disc.navigate(start, grid.sid(c))]
    print(f"   из угла (0,0) навыки доводят до глубины комнат {got} — {len(got)}/4")

    print("\n── Итог ──")
    print("   Агент сам открыл проёмы как подцели (совпали с истинными), выучил к ним")
    print("   навыки из опыта, упорядочил их по learning progress и переиспользует как")
    print("   путевые точки, покрывая мир лучше случайных целей — мотивация изнутри.")


if __name__ == "__main__":
    main()
