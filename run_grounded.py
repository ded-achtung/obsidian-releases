#!/usr/bin/env python3
"""Рассуждение, зазёмленное в мире: индукция правила из наблюдений агента.

Агент делает НЕСКОЛЬКО шагов, наблюдает (клетка, действие, след. клетка) и тем же
движком индукции выводит ПРАВИЛО динамики на каждое действие — из пары примеров, а
не таблицей. Выученной моделью он предсказывает переходы во всём мире и планирует
путь к цели. Это «учиться из малого», сомкнутое с восприятием/миром.

Запуск: python run_grounded.py
"""

from __future__ import annotations

from thinking_system.world.gridworld import GridWorld
from thinking_system.world.rooms import rooms_world
from thinking_system.reasoning.grounded import observe, induce_dynamics, WorldRule


def true_next(grid, s, a):
    return grid.move_from(s, GridWorld.MOVES[a])


def main():
    grid = rooms_world()
    free = grid.free_cells()
    print(f"▶ Агент в мире-комнатах {grid.size}×{grid.size}; правило динамики выводится ИЗ ЕГО наблюдений\n")

    steps = 20
    obs = observe(grid, steps, seed=0)
    rules = induce_dynamics(obs)
    print(f"1) НАБЛЮДЕНИЯ → ПРАВИЛО: агент прошёл {steps} шагов и вывел эффект каждого действия:")
    for a in sorted(rules):
        print(f"   действие {a} ({grid.ARROWS[a]}) → правило «{rules[a]}»  (из {sum(1 for s, aa, sp in obs if aa == a and sp != s)} примеров)")

    wr = WorldRule(grid, rules)
    tot = sum(1 for _ in free) * 4
    ok = sum(wr.predict(s, a) == true_next(grid, s, a) for s in free for a in range(4))
    seen = {(s, a) for s, a, _ in obs}
    print(f"\n2) ОБОБЩЕНИЕ модели на ВЕСЬ мир ({tot} пар клетка×действие):")
    print(f"   индукция правила:   {100 * ok / tot:.0f}% верных предсказаний (из ~5 наблюдений на действие)")
    print(f"   табличная память:   {100 * len(seen) / tot:.0f}% (знает лишь увиденные пары — остальное мимо)")

    print("\n3) ИСПОЛЬЗОВАНИЕ — планирование по ВЫУЧЕННОЙ модели:")
    path = wr.plan((0, 0), (6, 6))
    print(f"   путь (0,0)→(6,6) по выученной динамике: {len(path)} шагов {[grid.ARROWS[a] for a in path]}")

    print("\n── Итог ──")
    print("   Тот же движок индукции, но примеры теперь ИЗ МИРА: агент по горстке своих")
    print("   наблюдений вывел правило динамики, обобщил на все клетки и спланировал путь.")
    print("   Рассуждение сомкнуто с восприятием — модель из малого, а не таблица из многого.")


if __name__ == "__main__":
    main()
