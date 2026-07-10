#!/usr/bin/env python3
"""Mind над ТРЕТЬИМ типом опыта: миры-лабиринты в общем маршрутизаторе.

Сценарий (продолжение run_mind.py, теперь про действие):

  СЕССИЯ 1: один агент получает вперемешку ТЕКСТ и десять СВЕЖИХ лабиринтов
    из случайного распределения (как в run_world/run_transfer). Каждый опыт
    агент маршрутизирует сам: текст → читает, мир → исследует существующей
    машинерией активного вывода (ActingAgent: прагматика/эпистемика).
    Беглым взглядом (ступень 1 лестницы исследования) часть миров честно
    не решается — они попадают в ПОВЕСТКУ; агент сам возвращается и
    исследует дольше. Решённость мира проверяется ИСПОЛНЕНИЕМ маршрута
    в среде, длина сравнивается с истинным оптимумом (BFS по карте).

  СЕССИЯ 2: новый процесс загружает состояние — карты миров пережили
    запуск, известные миры решаются сразу, без исследования.

Честные границы: карта мира ПРО-лабиринтная, переноса между мирами нет
(это предмет run_transfer.py); мера здесь — единый агент маршрутизирует
ТРИ типа опыта, эскалирует исследование по собственной повестке и помнит.

Запуск: python run_mind_worlds.py
"""

from __future__ import annotations

import argparse
import os

from thinking_system.mind import Mind
from thinking_system.world.maze_dist import random_maze

TEXT = "Например, отражение превращает [[5, 0, 6]] в [[6, 0, 5]]."


def maze_item(seed: int) -> dict:
    env = random_maze(seed)
    return {"id": f"maze-{seed}",
            "world": {"size": env.size, "walls": sorted(env.walls),
                      "start": list(env.start), "goal": list(env.goal)}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", default="mind_worlds_state.json")
    ap.add_argument("--seeds", type=int, nargs=2, default=(200, 210),
                    metavar=("FROM", "TO"), help="сиды свежих лабиринтов [from, to)")
    args = ap.parse_args()

    if os.path.exists(args.state):
        os.remove(args.state)                                # честный старт с нуля
    worlds = [maze_item(s) for s in range(*args.seeds)]
    stream = [TEXT] + worlds                                 # текст и миры вперемешку
    print(f"▶ Единый агент • ТРИ типа опыта: текст + {len(worlds)} свежих лабиринтов 7×7\n")

    # ── СЕССИЯ 1 ─────────────────────────────────────────────────────────────────
    mind = Mind(args.state)
    print("СЕССИЯ 1 (беглый взгляд: ступень 1 лестницы исследования)")
    for item in stream:
        kind = mind.perceive(item)
        res = mind.experience(item, effort=1)
        if kind == "text":
            print(f"   текст → прочитан: выучено {res['выучено_слов']}")
        else:
            status = (f"дошёл за {res['steps']} шагов (оптимум {res['optimal']})"
                      if res["solved"] else "не дошёл — в повестку")
            print(f"   {item['id']} → мир: исследовано {res['explored']} шагов, {status}")
    solved_1 = len(worlds) - len(mind.unsolved_worlds)
    print(f"   итог беглого взгляда: решено {solved_1}/{len(worlds)}; "
          f"в повестке миров: {len(mind.unsolved_worlds)}")

    print(f"\nПовестка агента: {mind.agenda()}")
    work = mind.idle_work()
    for wid, steps, opt, explored in work["worlds_resolved"]:
        print(f"   вернулся и исследовал дольше {wid}: дошёл за {steps} шагов "
              f"(оптимум {opt}; исследовано ещё {explored})")
    ratios = []
    for item in worlds:
        res = mind.explore(item["id"], item["world"], effort=1)  # известные — сразу по карте
        if res["solved"]:
            ratios.append(res["steps"] / res["optimal"])
    n_ok = len(ratios)
    print(f"\nИтог сессии 1: дошёл в {n_ok}/{len(worlds)} мирах; длина маршрута "
          f"к оптимуму: ×{sum(ratios) / n_ok:.2f}" if n_ok else "ни один мир не решён")
    path = mind.save()
    print(f"Состояние сохранено: {path} (карты миров: {len(mind.world_maps)})")

    # ── СЕССИЯ 2: новый процесс, та же память ────────────────────────────────────
    print("\nСЕССИЯ 2 (новый агент, загрузил состояние)")
    mind2 = Mind(args.state)
    for item in worlds[:3]:
        res = mind2.explore(item["id"], item["world"], effort=1)
        print(f"   {item['id']}: дошёл за {res['steps']} шагов, исследовано "
              f"{res['explored']} — карта пережила процесс")

    print("\n── Итог (честно) ──")
    print("   Один агент маршрутизирует ТРИ типа опыта (текст / грид-задачи / миры);")
    print("   способность к мирам — существующий ActingAgent (активный вывод), карта")
    print("   мира — таблица переходов. Нерешённое беглым взглядом попадает в повестку,")
    print("   агент сам исследует дольше; решённость проверяется исполнением в среде;")
    print("   карты переживают процесс. Границы: карта про-лабиринтная, переноса")
    print("   МЕЖДУ мирами нет (предмет run_transfer.py); лабиринты — свежие из")
    print("   распределения, но игрушечные (7×7, полная наблюдаемость).")


if __name__ == "__main__":
    main()
