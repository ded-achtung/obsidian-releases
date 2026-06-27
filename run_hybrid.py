#!/usr/bin/env python3
"""Частичный успех рассуждения: правило для регулярного + заплатки на исключения.

Реальный мир редко целиком регулярен. Чистое правило «всё-или-ничего» рушится от
одной нерегулярной клетки. Гибрид выводит правило для БОЛЬШИНСТВА переходов
(обобщает на невиданные регулярные клетки) и запоминает немногие ИСКЛЮЧЕНИЯ —
стоимость по доле нерегулярного, а не по размеру мира.

Запуск: python run_hybrid.py
"""

from __future__ import annotations

import numpy as np

from thinking_system.world.gridworld import GridWorld
from thinking_system.agent.meta_agent import GridEnv
from thinking_system.reasoning.hybrid import HybridWorldModel
from thinking_system.reasoning.grounded import induce_dynamics, WorldRule


def main():
    grid = GridWorld(11, set(), start=(0, 0), goal=(10, 10))   # открытый 11×11 = 121 клетка
    free = [(r, c) for r in range(11) for c in range(11)]
    env = GridEnv(grid, scramble_seed=3, n_scramble=12)        # 12 нерегулярных клеток
    print(f"▶ Мир {len(free)} клеток, {len(env.perm)} перепутаны; пар клетка×действие: {len(free) * 4}\n")

    def walk(steps, seed=0):
        rng = np.random.default_rng(seed)
        s = grid.start
        obs = []
        for _ in range(steps):
            a = int(rng.integers(4)); sp = env.transition(s, a); obs.append((s, a, sp)); s = sp
        return obs

    def acc(model):
        return np.mean([model.predict(s, a) == env.transition(s, a) for s in free for a in range(4)])

    def tab_acc(obs):
        tab = {(s, a): sp for s, a, sp in obs}
        return np.mean([tab.get((s, a), s) == env.transition(s, a) for s in free for a in range(4)])

    print("1) ТОЧНОСТЬ МОДЕЛИ при фиксированном бюджете наблюдений:")
    print("   бюджет | чистое правило | ГИБРИД (правило+заплатки) | табличная память")
    for budget in [150, 400, 1000]:
        obs = walk(budget)
        hyb = HybridWorldModel(grid).fit(obs)
        pure = WorldRule(grid, induce_dynamics(obs))
        print(f"   {budget:>6} | {acc(pure):>13.0%} | {acc(hyb):>16.0%} ({len(hyb.exceptions):>2} заплаток) | {tab_acc(obs):>15.0%}")

    # обобщение: гибрид верен на регулярных клетках, которых НЕ видел
    obs = walk(400)
    hyb = HybridWorldModel(grid).fit(obs)
    seen = {(s, a) for s, a, _ in obs}
    regular = [(s, a) for s in free for a in range(4) if s not in env.perm and (s, a) not in seen]
    gen = np.mean([hyb.predict(s, a) == env.transition(s, a) for s, a in regular])
    print(f"\n2) ОБОБЩЕНИЕ: на {len(regular)} регулярных парах, которых агент НЕ видел — {gen:.0%} верно")
    print("   (правило переносит регулярное на невиданное; табличная память тут бессильна)")

    print("\n3) ПЛАНИРОВАНИЕ замкнутым циклом (план → шаг → доучить исключение → переспланировать):")
    before = len(hyb.exceptions)
    reached, steps = hyb.reach(env, (0, 0), (10, 10))
    print(f"   (0,0)→(10,10): {'дошёл' if reached else 'нет'} за {steps} шагов; "
          f"по пути доучил {len(hyb.exceptions) - before} исключений на сюрпризах")

    print("\n── Итог ──")
    print("   Не «или вывел правило, или учи всё»: агент выводит правило для регулярного")
    print("   большинства (и переносит его на невиданное), а немногие исключения запоминает.")
    print("   Частичное рассуждение — мост между дешёвой моделью и надёжной памятью.")


if __name__ == "__main__":
    main()
