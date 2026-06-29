#!/usr/bin/env python3
"""Объединение трёх расширений в ОДНОМ агенте:
восприятие сквозь шум (п.1) + вера под частичной наблюдаемостью (п.2) + язык (п.3).

Агент получает команду словами, действует в зашумлённом частично-наблюдаемом мире,
распознаёт обзор обученным восприятием, ведёт веру о позиции, локализуется и доходит
до заданной языком цели.

Запуск: python run_unified.py
"""

from __future__ import annotations

import numpy as np

from thinking_system.agent.unified import SoftmaxClassifier, UnifiedAgent
from thinking_system.language.grounding import GoalClassifier, BagOfWords, generate_commands
from thinking_system.world.noisy_partial import NoisyPartialWorld, unified_maze
from thinking_system.world.partial import local_pattern


def main():
    grid = unified_maze()
    size = grid.size
    free = [s for s in range(grid.n_states) if (s // size, s % size) not in grid.walls]
    patterns = sorted(set(local_pattern(grid, s) for s in free))
    pattern_classes = {p: i for i, p in enumerate(patterns)}
    dim = 16

    world = NoisyPartialWorld(grid, dim=dim, noise=0.5, seed=0)
    print(f"▶ Зашумлённый частично наблюдаемый мир {size}×{size}; наблюдение = обзор+шум (dim={dim})")
    print(f"  {len(free)} клеток, {len(patterns)} типов локального обзора\n")

    # обучаем ВОСПРИЯТИЕ: шумный вектор → класс обзора
    X, y = [], []
    for s in free:
        cls = pattern_classes[local_pattern(grid, s)]
        for _ in range(40):
            X.append(world.observe_at(s)); y.append(cls)
    perception = SoftmaxClassifier(dim, len(patterns))
    perception.fit(np.array(X), np.array(y), epochs=300)
    test_acc = np.mean([perception.predict(world.observe_at(s)[None])[0] == pattern_classes[local_pattern(grid, s)] for s in free for _ in range(20)])
    print(f"1) ВОСПРИЯТИЕ: распознаёт обзор сквозь шум с точностью {test_acc * 100:.0f}%")

    agent = UnifiedAgent(grid, perception, pattern_classes, seed=0)
    rng = np.random.default_rng(0)
    commands = [
        ("go to the top right corner", 1),
        ("head to the bottom left", 2),
        ("navigate to the center", 4),
        ("reach the upper left", 0),
        ("walk to the lower right corner", 3),
    ]

    # обучаем ЯЗЫК на командах БЕЗ тестовых (честный held-out: тест-команды не видны при обучении;
    # словарь закрытый — тест = новые комбинации знакомых слов)
    test_texts = {t for t, _ in commands}
    train_cmds = [c for c in generate_commands() if c[0] not in test_texts]
    goal_clf = GoalClassifier(BagOfWords([t for t, _ in train_cmds]))
    goal_clf.fit(train_cmds, epochs=300)

    print("\n2) КОМАНДА → (восприятие+вера) → ДЕЙСТВИЕ под шумом и частичной наблюдаемостью:")
    print(f"   {'команда':<32}{'цель':<14}{'итог':>22}")
    ok_total = 0
    for text, expected in commands:
        cls, cell = agent.set_goal_from_text(text, goal_clf, size)
        grid.goal = cell  # мир сообщает «дошёл» именно по понятой языком цели
        goal_sid = grid.sid(cell)
        starts = [s for s in free if s != goal_sid]
        succ, steps_sum = 0, 0
        for trial in range(8):
            start = int(rng.choice(starts))
            agent.b = np.ones(agent.F) / agent.F
            agent.observe_noisy(world.reset(start))
            for t in range(1, 201):
                a = agent.act()
                obs, done = world.step(a)
                agent.predict(a)
                agent.observe_noisy(obs)
                if done:
                    succ += 1
                    steps_sum += t
                    break
        ok = cls == expected
        ok_total += ok
        avg = steps_sum / max(succ, 1)
        from thinking_system.language.grounding import PLACES
        print(f"   {text:<32}{PLACES[cls][0]:<14}{f'{succ}/8 дошёл, ~{avg:.0f} шаг':>22}  {'✓' if ok else '✗'}")

    print(f"\n   язык понят верно: {ok_total}/{len(commands)} команд (held-out: тест-команды не было при обучении)")
    print("\n── Итог ──")
    print("   Один агент: понимает команду словами, распознаёт зашумлённый обзор обученным")
    print("   восприятием, ведёт веру о позиции под частичной наблюдаемостью и доходит до")
    print("   заданной языком цели — три расширения работают вместе.")


if __name__ == "__main__":
    main()
