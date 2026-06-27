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
    r, c = s
    dr, dc = GridWorld.MOVES[a]
    nr, nc = r + dr, c + dc
    return (nr, nc) if 0 <= nr < grid.size and 0 <= nc < grid.size and (nr, nc) not in grid.walls else s


def main():
    grid = rooms_world()
    free = [(s // grid.size, s % grid.size) for s in range(grid.n_states) if (s // grid.size, s % grid.size) not in grid.walls]
    print(f"▶ Агент в мире-комнатах {grid.size}×{grid.size}; правило динамики выводится ИЗ ЕГО наблюдений\n")

    # наблюдаем, пока не увидим эффект всех 4 действий (иначе правило неполно при
    # коротком блуждании — индукции нужен ≥1 пример движения на каждое действие)
    steps = 20
    obs = observe(grid, steps, seed=0)
    rules = induce_dynamics(obs)
    while len(rules) < 4 and steps < 200:
        steps += 10
        obs = observe(grid, steps, seed=0)
        rules = induce_dynamics(obs)
    print(f"1) НАБЛЮДЕНИЯ → ПРАВИЛО: агент прошёл {steps} шагов и вывел эффект каждого действия:")
    for a in sorted(rules):
        print(f"   действие {a} ({grid.ARROWS[a]}) → правило «{rules[a]}»  (из {sum(1 for s, aa, sp in obs if aa == a and sp != s)} примеров)")

    wr = WorldRule(grid, rules)
    tot = sum(1 for _ in free) * 4
    ok = sum(wr.predict(s, a) == true_next(grid, s, a) for s in free for a in range(4))
    # честный бейзлайн «табличная память»: для увиденных пар — запомненный исход,
    # для невиданных — разумный дефолт «остался на месте»; меряем ТУ ЖЕ точность предсказания
    table = {(s, a): sp for s, a, sp in obs}
    ok_table = sum(table.get((s, a), s) == true_next(grid, s, a) for s in free for a in range(4))
    print(f"\n2) ОБОБЩЕНИЕ модели на ВЕСЬ мир ({tot} пар клетка×действие):")
    print(f"   индукция правила:   {100 * ok / tot:.0f}% верных предсказаний (из ~5 наблюдений на действие)")
    print(f"   табличная память:   {100 * ok_table / tot:.0f}% (помнит увиденные пары, на остальных дефолт «стоять»)")

    print("\n3) ИСПОЛЬЗОВАНИЕ — планирование по ВЫУЧЕННОЙ модели:")
    path = wr.plan((0, 0), (6, 6))
    print(f"   путь (0,0)→(6,6) по выученной динамике: {len(path)} шагов {[grid.ARROWS[a] for a in path]}")

    print("\n── Итог ──")
    print("   Тот же движок индукции, но примеры теперь ИЗ МИРА: агент по горстке своих")
    print("   наблюдений вывел правило динамики, обобщил на все клетки и спланировал путь.")
    print("   Рассуждение сомкнуто с восприятием — модель из малого, а не таблица из многого.")


if __name__ == "__main__":
    main()
