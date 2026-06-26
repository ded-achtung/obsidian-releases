#!/usr/bin/env python3
"""Язык ВЫБИРАЕТ выученный навык: команда → проём → Q-опция → исполнение.

Сначала агент Q-learning выучивает 4 опции-навыка (к проёмам мира-комнат). Затем
обучается грунтинг команд в проёмы. На НЕВИДАННЫХ формулировках команда выбирает
нужную опцию, и агент доходит. Цепочка команд → цепочка опций (иерархия из языка).

Запуск: python run_language_options.py
"""

from __future__ import annotations

import numpy as np

from thinking_system.world.rooms import rooms_world
from thinking_system.agent.qoption import QOptionLibrary
from thinking_system.agent.language_options import LanguageOptionAgent
from thinking_system.language.doorways import train_doorway_classifier, accuracy, DOORWAYS


def main():
    grid = rooms_world()
    free = [s for s in range(grid.n_states) if (s // grid.size, s % grid.size) not in grid.walls]
    doorways = [grid.sid(DOORWAYS[i][0]) for i in range(4)]
    print(f"▶ Мир-комнаты {grid.size}×{grid.size}; 4 опции-навыка к проёмам + грунтинг команд\n")

    # 1) выучить опции из опыта (Q-learning)
    lib = QOptionLibrary(grid, doorways, seed=0)
    lib.train(episodes_each=1500)

    # 2) выучить язык: команда → проём (проверка на невиданных формулировках)
    clf, train, held = train_doorway_classifier(seed=0)
    print(f"1) ГРУНТИНГ ЯЗЫКА команда → проём: обучен на {len(train)} фразах")
    print(f"   точность на НЕВИДАННЫХ формулировках (обобщение): {accuracy(clf, held):.0%}")

    agent = LanguageOptionAgent(lib, clf)

    # 3) команды (в т.ч. незнакомые формулировки) выбирают опцию и доходят
    print("\n2) КОМАНДА → ВЫБОР ОПЦИИ → ИСПОЛНЕНИЕ (старт из угла (0,0)):")
    start = grid.sid((0, 0))
    name = {i: DOORWAYS[i][1] for i in range(4)}
    demo_cmds = [
        "take the northern passage", "head to the lower gap",
        "navigate to the western door", "go through the rightmost opening",
    ]
    for cmd in demo_cmds:
        r = agent.obey(cmd, start, seed=0)
        print(f"   «{cmd:<34}» → проём «{name[r['doorway']]:<6}» ({r['confidence']:.0%}) — {'дошёл' if r['reached'] else 'нет'}")

    # 4) на всех held-out командах: верный ли навык выбран и доходит ли он
    succ = corr = 0
    rng = np.random.default_rng(1)
    for cmd, y in held:
        r = agent.obey(cmd, int(rng.choice(free)), seed=int(rng.integers(10 ** 6)))
        corr += int(r["doorway"] == y)
        succ += int(r["reached"] and r["doorway"] == y)
    print(f"\n   на {len(held)} невиданных командах: верный навык {100 * corr / len(held):.0f}%, "
          f"довёл до проёма {100 * succ / len(held):.0f}%")

    # 5) цепочка команд → цепочка опций (иерархия из языка)
    print("\n3) ЦЕПОЧКА КОМАНД → ЦЕПОЧКА ОПЦИЙ (из угла через два проёма):")
    seq_cmds = ["go through the upper doorway", "reach the eastern passage"]
    r = agent.obey_sequence(seq_cmds, start, seed=0)
    chain = " → ".join(name[i] for i in range(4) if grid.sid(DOORWAYS[i][0]) in r["subgoals"])
    print(f"   [{seq_cmds[0]}] + [{seq_cmds[1]}]  ⇒  {chain}: {'дошёл' if r['reached'] else 'нет'}")

    print("\n── Итог ──")
    print("   Язык понимает НЕВИДАННЫЕ формулировки и выбирает нужный ВЫУЧЕННЫЙ навык;")
    print("   поведение — композиция готовых опций, а цепочка команд строит маршрут")
    print("   на верхнем уровне. «Как назвать — так и сделать».")


if __name__ == "__main__":
    main()
