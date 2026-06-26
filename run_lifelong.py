#!/usr/bin/env python3
"""Долгоживущий автономный агент: со временем умнеет, накапливая навыки.

Единый контур в одном существе: исследует → СПЕКТРАЛЬНО открывает подцели →
учит навыки ИЗ ВОСПРИЯТИЯ (Q над латентами JEPA) → принимает команды на ЯЗЫКЕ и
исполняет. Жизнь идёт фазами: библиотека навыков растёт, покрытие мира и доля
выполнимых команд поднимаются.

Запуск: python run_lifelong.py
"""

from __future__ import annotations

import numpy as np

from thinking_system.world.rooms import rooms_world
from thinking_system.world.gridworld import GridWorld
from thinking_system.world.features import CellFeatures
from thinking_system.world.latent_model import LatentWorldModel
from thinking_system.agent.latent_qoption import jepa_encoder
from thinking_system.agent.lifelong import LifelongAgent


def train_perception(grid, rng):
    """«Рождение»: выучить восприятие (JEPA) на первом опыте мира."""
    feats = CellFeatures(grid.n_states, dim=24, noise=0.15, seed=0)
    jepa = LatentWorldModel(24, 4, latent_dim=16, lr=5e-4, var_coef=0.02, seed=0)

    def move(s, a):
        r, c = divmod(s, grid.size)
        dr, dc = GridWorld.MOVES[a]
        nr, nc = r + dr, c + dc
        return grid.sid((nr, nc)) if 0 <= nr < grid.size and 0 <= nc < grid.size and (nr, nc) not in grid.walls else s

    O, A, NO, s = [], [], [], grid.sid(grid.start)
    for _ in range(10000):
        a = int(rng.integers(4)); sp = move(s, a)
        O.append(feats.observe(s)); A.append(a); NO.append(feats.observe(sp)); s = sp
    O, A, NO = np.array(O), np.array(A), np.array(NO)
    for _ in range(60):
        ii = rng.permutation(len(O))
        for k in range(0, len(O), 128):
            b = ii[k:k + 128]; jepa.update(O[b], A[b], NO[b])
    return jepa_encoder(jepa, feats.observe, normalize=True), jepa.Ld + 1


def main():
    grid = rooms_world()
    rng = np.random.default_rng(0)
    print(f"▶ Один долгоживущий агент в мире-комнатах {grid.size}×{grid.size}\n")

    print("«Рождение»: учим восприятие (JEPA) на первом опыте…")
    encode, nf = train_perception(grid, rng)
    agent = LifelongAgent(grid, encode, nf, alpha=0.1, seed=0)

    print("\nЖИЗНЬ ПО ФАЗАМ (за фазу: добрать опыт → открыть горлышки → освоить новый навык):")
    print("  фаза | опыт(шагов) | навыков | покрытие мира | команд выполнимо")
    for ph in range(1, 5):
        st = agent.live_phase(explore_steps=3000, episodes=7000, max_new=1)
        print(f"   {ph:>3} | {st['experience']:>10} | {st['skills']:>6}/4 | {st['coverage']:>11.0%} | {st['obey']:>13.0%}")

    print("\nИСПОЛНЕНИЕ КОМАНД зрелым агентом (язык → навык из библиотеки → действие):")
    free = agent.disc.free
    for cmd in ["take the northern passage", "head to the lower gap",
                "navigate to the western door", "go through the rightmost opening"]:
        rate = np.mean([agent.obey(cmd, int(rng.choice(free)), seed=int(rng.integers(10 ** 6)))["reached"] for _ in range(12)])
        print(f"   «{cmd:<34}» → исполнено {rate:.0%} (из 12 случайных стартов)")

    # обобщение языка на невиданных формулировках
    succ = sum(agent.obey(c, int(rng.choice(agent.disc.free)), seed=int(rng.integers(10 ** 6)))["reached"]
               for c, _ in agent.held[:120]) / 120
    print(f"\n   на 120 НЕВИДАННЫХ командах исполнено: {succ:.0%}")

    print("\n── Итог ──")
    print("   Один агент прожил жизнь: выучил восприятие, сам открыл подцели из спектра")
    print("   опыта, нарастил библиотеку навыков из восприятия и научился исполнять")
    print("   команды на языке. Чем дольше живёт — тем способнее. Весь контур замкнут.")


if __name__ == "__main__":
    main()
