#!/usr/bin/env python3
"""П.3: навыки как ПОЛИТИКИ (опции) vs фиксированные маршруты — устойчивость к сбоям.

Сравнивает достижение цели при возмущениях (агента случайно сбивает с пути):
маршрут-навык слепо выполняет действия и ломается; навык-опция (политика) перенаводит
из любой клетки и доходит.

Запуск: python run_skills.py
"""

from __future__ import annotations

import numpy as np

from thinking_system.memory.hierarchical import HierarchicalMemory
from thinking_system.memory.skill_policies import OptionPolicies
from thinking_system.world.rooms import rooms_world


def main():
    grid = rooms_world()
    free = grid.free_sids()
    mem = HierarchicalMemory(grid)
    mem.explore(20000, seed=0)
    mem.consolidate(n_landmarks=4)
    opt = OptionPolicies(grid, mem)

    print("▶ Мир-комнаты; навыки-маршруты vs навыки-политики (опции) при сбоях\n")
    print(f"   {'сбой/шаг':>10}{'маршрут дошёл':>16}{'опция (политика) дошла':>26}")
    rng = np.random.default_rng(1)
    n = 80
    for perturb in [0.0, 0.08, 0.16]:
        route = opt_ok = 0
        for _ in range(n):
            start, goal = (int(x) for x in rng.choice(free, 2, replace=False))
            seed = int(rng.integers(10 ** 6))
            route += opt.navigate_route(start, goal, perturb=perturb, seed=seed)
            opt_ok += opt.navigate_options(start, goal, perturb=perturb, seed=seed)
        print(f"   {perturb:>10.2f}{f'{100 * route / n:.0f}%':>16}{f'{100 * opt_ok / n:.0f}%':>26}")

    print("\n── Итог ──")
    print("   Навык-ПОЛИТИКА (опция) ведёт к ориентиру из любой точки и восстанавливается")
    print("   после сбоя; фиксированный маршрут ломается. Это иерархический RL: высокий")
    print("   уровень компонует опции, низкий — устойчиво их исполняет.")


if __name__ == "__main__":
    main()
