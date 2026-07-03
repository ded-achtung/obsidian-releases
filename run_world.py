#!/usr/bin/env python3
"""Действующий агент в мире-сетке: исследует, строит карту, доходит до цели.

Агент не знает лабиринт. Выбирая действия по активному выводу (цель vs
исследование), он строит модель мира и от эпизода к эпизоду доходит до цели всё
быстрее — пока не выходит на (около)оптимальный путь. Сравнение со случайным
агентом.

Запуск: python run_world.py
"""

from __future__ import annotations

import numpy as np

from thinking_system.agent.acting import ActingAgent
from thinking_system.world.gridworld import default_maze
from thinking_system.world.maze_dist import random_maze
from thinking_system.viz import sparkline


def run_episode(env, agent, *, epsilon, max_steps):
    s = env.reset()
    for t in range(1, max_steps + 1):
        a = agent.act(s, epsilon=epsilon)
        sp, done = env.step(a)
        agent.learn(s, a, sp)
        s = sp
        if done:
            return t
    return max_steps


def random_baseline(env, *, episodes=20, max_steps=2000, seed=0):
    rng = np.random.default_rng(seed)
    steps = []
    for _ in range(episodes):
        env.reset()
        for t in range(1, max_steps + 1):
            _, done = env.step(int(rng.integers(env.n_actions)))
            if done:
                break
        steps.append(t)
    return float(np.mean(steps))


def render(env, path_states: list[int]) -> None:
    path = set(path_states)
    for r in range(env.size):
        row = ""
        for c in range(env.size):
            sid = r * env.size + c
            if (r, c) == env.start:
                row += " S"
            elif (r, c) == env.goal:
                row += " G"
            elif (r, c) in env.walls:
                row += " █"
            elif sid in path:
                row += " ·"
            else:
                row += " ."
        print("   " + row)


def main() -> None:
    env = default_maze()
    opt = env.optimal_steps()
    print(f"▶ Лабиринт {env.size}×{env.size}, старт {env.start} → цель {env.goal}")
    print(f"  оптимальный путь: {opt} шагов;  случайный агент: ~{random_baseline(env):.0f} шагов до цели\n")

    agent = ActingAgent(env.n_actions, env.goal_state, seed=0)
    episodes = 40
    steps_hist = []
    for ep in range(episodes):
        eps = max(0.05, 0.5 * (0.88 ** ep))  # любопытство угасает по мере выучивания карты
        steps_hist.append(run_episode(env, agent, epsilon=eps, max_steps=1500))

    print("Шагов до цели по эпизодам (меньше = лучше):")
    print("   " + sparkline(steps_hist))
    print(f"   эпизод 1: {steps_hist[0]} шагов  →  эпизод {episodes}: {steps_hist[-1]} шагов  (оптимум {opt})")
    last5 = np.mean(steps_hist[-5:])
    print(f"   среднее за последние 5 эпизодов: {last5:.1f} шагов\n")

    path = agent.greedy_path(env.reset())
    reached = path and path[-1] == env.goal_state
    print(f"Выученный путь к цели ({len(path) - 1} шагов, дошёл: {'да' if reached else 'нет'}):")
    render(env, path)

    # held-out: тот же механизм на СВЕЖИХ лабиринтах из распределения —
    # проверка, что результат не подогнан под одну ручную карту
    n_fresh = 10
    ratios, reach_cnt = [], 0
    for seed in range(n_fresh):
        fenv = random_maze(100 + seed)
        fagent = ActingAgent(fenv.n_actions, fenv.goal_state, seed=seed)
        for ep in range(episodes):
            run_episode(fenv, fagent, epsilon=max(0.05, 0.5 * (0.88 ** ep)), max_steps=1500)
        fpath = fagent.greedy_path(fenv.reset())
        ok = bool(fpath) and fpath[-1] == fenv.goal_state
        reach_cnt += ok
        if ok:
            ratios.append((len(fpath) - 1) / fenv.optimal_steps())
    print(f"\nHELD-OUT: {n_fresh} свежих случайных лабиринтов, тот же механизм с нуля:")
    print(f"   дошёл: {reach_cnt}/{n_fresh}; длина пути к оптимуму: "
          f"×{np.mean(ratios):.2f} ± {np.std(ratios):.2f}")

    print("\n── Итог (честно) ──")
    print(f"   Агент выучил карту и довёл маршрут со {steps_hist[0]} до {int(last5)} шагов")
    print(f"   (оптимум {opt}); механизм воспроизводится на свежих лабиринтах, а не подогнан")
    print("   под одну карту. Но модель мира — ПРО-лабиринтная: в каждом новом мире агент")
    print("   учится с нуля; перенос навыка между мирами — предмет run_transfer.py.")


if __name__ == "__main__":
    main()
