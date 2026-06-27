#!/usr/bin/env python3
"""Мета-агент: рассуждать, когда можно; учиться методом проб, когда нужно.

Один агент смыкает оба капстоуна и САМ выбирает путь: пробует вывести правило мира
(дёшево), а если правило мир не описывает — откатывается к обучению из опыта
(дорого, но надёжно). Открытие подцелей и язык — общие. Один и тот же агент в двух
мирах: обычном (рассуждает) и «перепутанном» (вынужден тренироваться).

Запуск: python run_meta_agent.py
"""

from __future__ import annotations

from thinking_system.world.rooms import rooms_world
from thinking_system.agent.meta_agent import GridEnv, MetaAgent

TRUE_DOORS = {(1, 3), (5, 3), (3, 1), (3, 5)}


def run(title, env, *, seed=0):
    agent = MetaAgent(env, seed=seed)
    r = agent.live(rl_steps=4000, rl_episodes=1500)
    d = agent.diag
    print(f"{title}")
    print(f"   проверка правила: выведено {d['rules']}/4 действий, точность на свежих наблюдениях {d['verify_acc']:.0%}")
    print(f"   ВЫБРАН РЕЖИМ: {r['mode']}")
    print(f"   опыт потрачено: {r['cost']} переходов")
    print(f"   подцели найдены: {len(set(r['subgoals']) & TRUE_DOORS)}/4 проёма   |   языковые команды: {r['commands']:.0%}")
    return agent


def main():
    print("▶ Один мета-агент в двух мирах: рассуждать или тренироваться — выбор по проверке\n")

    run("МИР A — обычный (динамика выразима простым правилом):", GridEnv(rooms_world()))
    print()
    run("МИР B — перепутанная динамика (в каждой клетке свой смысл действий):",
        GridEnv(rooms_world(), scramble_seed=1))

    print("\n── Итог ──")
    print("   Тот же агент в мире A вывел правило и решил почти даром (десятки шагов); в мире B")
    print("   правило не прошло проверку — и он честно откатился к обучению методом проб (тысячи")
    print("   шагов), всё равно открыв подцели и выполнив команды. Дёшево думать, когда")
    print("   получается; надёжно учиться, когда нет — арбитраж model-based ↔ model-free.")


if __name__ == "__main__":
    main()
