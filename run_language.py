#!/usr/bin/env python3
"""П.3: цель задаётся ЯЗЫКОМ — агент понимает команду и достигает цели.

Классификатор грунтит текстовую команду в пространственную цель. Обучаем на одних
КОМБИНАЦИЯХ слов, проверяем на ДРУГИХ (и на свежих, написанных вручную). Честно:
словарь ЗАКРЫТЫЙ (~30 слов) — «невиданные формулировки» это невиданные комбинации
знакомых слов, а не новые слова/синонимы. Это надёжная рекомбинация в закрытом
словаре, не понимание смысла слов. Затем действующий агент идёт к понятой цели.

Запуск: python run_language.py
"""

from __future__ import annotations

import numpy as np

from thinking_system.agent.acting import ActingAgent
from thinking_system.language.grounding import BagOfWords, GoalClassifier, PLACES, generate_commands, goal_cell
from thinking_system.world.gridworld import GridWorld


def navigate(size, goal, *, start=(3, 0), episodes=30):
    env = GridWorld(size, set(), start=start, goal=goal)
    agent = ActingAgent(4, env.goal_state, seed=0)
    steps = 0
    for ep in range(episodes):
        eps = max(0.05, 0.5 * (0.85 ** ep))
        s = env.reset()
        for steps in range(1, 401):
            a = agent.act(s, epsilon=eps)
            sp, done = env.step(a)
            agent.learn(s, a, sp)
            s = sp
            if done:
                break
    return steps, env.optimal_steps()


def main():
    data = generate_commands()
    rng = np.random.default_rng(0)
    rng.shuffle(data)
    cut = int(0.7 * len(data))
    train, test = data[:cut], data[cut:]

    bow = BagOfWords([t for t, _ in train])
    clf = GoalClassifier(bow)
    clf.fit(train)
    print(f"▶ Обучено на {len(train)} командах ({bow.size} слов в словаре, {len(PLACES)} целей)\n")

    acc_test = np.mean([clf.predict(t)[0] == c for t, c in test])
    print(f"1) Невиданные КОМБИНАЦИИ слов (held-out, закрытый словарь, {len(test)} команд): {acc_test * 100:.0f}%")
    print("   (held-out = новые сочетания знакомых слов, не новые слова — см. границы в docstring)")

    fresh = [
        ("please head over to the upper right", 1),
        ("walk to the lower left corner", 2),
        ("get to the middle", 4),
        ("proceed to the north west", 0),
        ("move to the south east", 3),
        ("navigate into the centre", 4),
    ]
    acc_fresh = np.mean([clf.predict(t)[0] == c for t, c in fresh])
    print(f"   на свежих фразах, написанных вручную ({len(fresh)}): {acc_fresh * 100:.0f}%\n")

    size = 7
    print("2) КОМАНДА → ПОНЯТАЯ ЦЕЛЬ → ДЕЙСТВИЕ (агент идёт к цели):")
    print(f"   {'команда':<38}{'понял как':<14}{'дошёл':>12}")
    for text, expected in fresh:
        cls, conf = clf.predict(text)
        cell = goal_cell(cls, size)
        steps, opt = navigate(size, cell)
        mark = "✓" if cls == expected else "✗"
        print(f"   {text:<38}{PLACES[cls][0]:<14}{f'{steps}/{opt} шаг':>12}  {mark}")

    print("\n── Итог ──")
    print("   Цель задаётся словами. Агент грунтит команду в пространственную цель и")
    print("   достигает её действием — связка ЯЗЫК → ЦЕЛЬ → ПОВЕДЕНИЕ. Обобщение — на")
    print("   невиданные комбинации слов закрытого словаря (не на новые слова/синонимы).")


if __name__ == "__main__":
    main()
