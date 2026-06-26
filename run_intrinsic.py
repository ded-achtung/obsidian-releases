#!/usr/bin/env python3
"""П.3: внутренняя мотивация — агент САМ ставит себе цели (без награды/языка).

Интринсик-агент выбирает цели на фронтире известного и расширяет компетенцию.
Сравнение со случайными самоцелями. Видно автокуррикулум (цели всё дальше) и более
быстрое освоение мира.

Запуск: python run_intrinsic.py
"""

from __future__ import annotations

import numpy as np

from thinking_system.agent.intrinsic import IntrinsicAgent
from thinking_system.world.gridworld import default_maze
from thinking_system.viz import sparkline


def run(grid, intrinsic, episodes, seed=0):
    ag = IntrinsicAgent(grid, intrinsic=intrinsic, seed=seed)
    reach, goal_dist = [], []
    for _ in range(episodes):
        r = ag.episode()
        reach.append(r["reachable"])
        goal_dist.append(r["goal_dist"])
    return ag, reach, goal_dist


def first_full(reach, n_free):
    for i, r in enumerate(reach):
        if r >= n_free:
            return i + 1
    return -1


def main():
    grid = default_maze()
    n_free = len([s for s in range(grid.n_states) if (s // grid.size, s % grid.size) not in grid.walls])
    episodes = 200
    print(f"▶ Лабиринт {grid.size}×{grid.size}, {n_free} клеток; агент сам ставит себе цели\n")

    ai, ri, gi = run(grid, True, episodes)
    ar, rr, gr = run(grid, False, episodes)

    print("1) ОСВОЕНИЕ МИРА — достижимых клеток по эпизодам (компетенция):")
    idx = np.linspace(0, episodes - 1, 50).astype(int)
    print("   интринсик: " + sparkline([ri[i] for i in idx]) + f"   полностью к эпизоду {first_full(ri, n_free)}")
    print("   случайно:  " + sparkline([rr[i] for i in idx]) + f"   полностью к эпизоду {first_full(rr, n_free)}")

    print("\n2) АВТОКУРРИКУЛУМ — сложность самопоставленных целей (дистанция) у интринсика:")
    early = np.mean([d for d in gi[:5] if d > 0])
    plateau = np.mean([d for d in gi[15:60] if d > 0])
    print("   " + sparkline(gi[:60]))
    print(f"   дистанция самоцели: первые эпизоды {early:.1f} → дальше {plateau:.1f}  (близкие цели → далёкие)")

    print("\n── Итог ──")
    print("   Без внешней цели агент сам формирует задачи на фронтире своих знаний,")
    print("   осваивает мир быстрее случайных самоцелей и выстраивает куррикулум")
    print("   от близких целей к далёким — внутренняя мотивация как двигатель.")


if __name__ == "__main__":
    main()
