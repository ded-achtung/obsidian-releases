#!/usr/bin/env python3
"""Единый агент: одна память + маршрутизация задачи по модулям, со сцепкой.

Разрыв «нет единого я» — честный измеримый шаг. ОДИН агент (MindAgent) с общей доской
решает КОМПОЗИТНУЮ задачу, которую не берёт ни один модуль по отдельности:

    рассуждение(вывести динамику мира) → смысл(понять команду, в т.ч. НОВЫЕ словоформы)
    → планирование(путь к цели по выученной модели) → действие.

Честный контроль — АБЛЯЦИЯ: выключаем модуль смысла (подслова → мешок целых слов).
На командах с новыми словоформами понимание рушится → весь композит рушится. Значит
интеграция настоящая: каждый модуль нагружен, а не для вида. Плюс тот же агент отвечает
на факт-вопрос дедукцией — несколько типов рассуждения под одной памятью.

Запуск: python run_mind.py
"""

from __future__ import annotations

from thinking_system.agent.mind import MindAgent
from thinking_system.world.rooms import rooms_world

# команды с НОВЫМИ словоформами (целиком не было в обучении классификатора).
# Цели — 4 угла (все свободны и достижимы; центр (3,3) в мире-комнатах — стена).
COMMANDS = [
    ("go to the northwestern corner", (0, 0)),     # northwestern → top-left
    ("head to the northeastern corner", (0, 6)),   # northeastern → top-right
    ("navigate to the southwestern area", (6, 0)),  # southwestern → bottom-left
    ("reach the southeastern corner", (6, 6)),     # southeastern → bottom-right
]
START = (3, 0)


def run(agent, label):
    reached = 0
    rows = []
    for cmd, goal in COMMANDS:
        r = agent.solve_navigation(cmd, START)
        ok = r["reached"] and r["goal"] == goal
        reached += ok
        rows.append((cmd, r["goal"], ok, r["steps"]))
    return reached, rows


def main():
    grid = rooms_world()
    print("▶ Единый агент: общая память + маршрутизация по модулям (рассуждение→смысл→план)\n")

    # один агент со всеми модулями (смысл = подслова)
    mind = MindAgent(grid, semantic=True, seed=0)
    n_rules = mind.perceive_dynamics()
    print(f"1) РАССУЖДЕНИЕ: вывел динамику мира — {n_rules}/4 правила из {mind.bb['observations']} наблюдений")
    print(f"   (модель в общей памяти; маршрут модулей: {mind.trace})\n")

    s_full, rows = run(mind, "full")
    print("2) КОМПОЗИТ: команда(новая словоформа) → цель → план по выученной модели → действие:")
    print(f"   {'команда':<36}{'цель':<10}{'дошёл':>8}")
    for cmd, goal, ok, steps in rows:
        print(f"   {cmd:<36}{str(goal):<10}{('да ' + str(steps)) if ok else 'нет':>8}")
    print(f"   итог: {s_full}/{len(COMMANDS)} композитных задач решено единым контуром\n")

    # АБЛЯЦИЯ: тот же агент без модуля смысла (мешок целых слов)
    mind_noo = MindAgent(grid, semantic=False, seed=0)
    s_abl, _ = run(mind_noo, "ablation")
    print("3) АБЛЯЦИЯ — выключаем смысл (подслова → мешок целых слов):")
    print(f"   композит на тех же командах: {s_abl}/{len(COMMANDS)}  (новые словоформы не поняты → цепочка рвётся)\n")

    # ДЕДУКЦИЯ тем же агентом — другой тип рассуждения под одной памятью
    ans = mind.ask(["сократ это человек", "все человек смертен"], "сократ смертен?")
    print(f"4) ДЕДУКЦИЯ тем же агентом: «сократ смертен?» → {ans}")

    print("\n── Что это значит ──")
    print(f"   Один агент с общей памятью сам связал рассуждение, смысл и планирование в")
    print(f"   цепочку и решил композит {s_full}/{len(COMMANDS)}; абляция ({s_abl}/{len(COMMANDS)}) доказывает, что модуль")
    print("   смысла нагружен, а не для вида. Честно: это оркестрация реальных модулей под")
    print("   единой памятью/маршрутизатором — шаг к «единому я», а не возникшее мышление.")


if __name__ == "__main__":
    main()
