#!/usr/bin/env python3
"""Эксперимент: пробить потолок DSL, ИЗОБРЕТАЯ примитив из данных задачи.

Фиксированный DSL (перебор 23 примитивов) решал 26/1000 — потолок: операции
беспараметрические. Здесь система СИНТЕЗИРУЕТ операцию из пар вход→выход конкретной
задачи (перекраска, масштаб, замощение), проверяет на всех обучающих парах и
применяет к тесту. Меряем: сколько решает «изобретение из данных» и сколько из них
НОВЫХ — задач, которых фиксированный DSL взять не мог.

Запуск: python run_arc_invent.py [--split train|eval]
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np
import arckit

from thinking_system.reasoning.grids import to_grid
from thinking_system.reasoning.invent import invent

# 26 задач, решённых фиксированным DSL (из ARC_RESULTS.md) — чтобы считать НОВЫЕ
DSL_SOLVED = {
    "007bbfb7", "0c786b71", "1cf80156", "1e0a9b12", "1f85a75f", "3906de3d", "3af2c5a8",
    "3c9b0459", "4347f46a", "445eab21", "496994bd", "5751f35e", "5b6cbef5", "6150a2bd",
    "62c24649", "67a3c6ac", "67e8384a", "68b16354", "7468f01a", "74dd1130", "8f2ea7aa",
    "9ddd00f0", "9dfd6313", "b8825c91", "be94b721", "ed36ccf7",
}


def task_grids(task):
    g = lambda a: to_grid(np.asarray(a).tolist())
    return [(g(i), g(o)) for i, o in task.train], [(g(i), g(o)) for i, o in task.test]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "eval"], default="train")
    args = p.parse_args()
    train_set, eval_set = arckit.load_data()
    tasks = list(train_set if args.split == "train" else eval_set)

    print(f"▶ Эксперимент: ИЗОБРЕТЕНИЕ примитива из данных, ARC {args.split} ({len(tasks)} задач)\n")
    start = time.time()
    by_inv = Counter()
    solved, new_solved = [], []
    for task in tasks:
        train, test = task_grids(task)
        name, fn = invent(train)
        if fn is None:
            continue
        try:
            ok = all(fn(ti) == to for ti, to in test)        # точное совпадение тестового выхода
        except Exception:  # noqa: BLE001
            ok = False
        if ok:
            solved.append((task.id, name))
            by_inv[name] += 1
            if task.id not in DSL_SOLVED:
                new_solved.append((task.id, name))

    elapsed = time.time() - start
    print(f"── Результат за {elapsed:.0f}с ──")
    print(f"   решено ИЗОБРЕТЕНИЕМ (точный тест): {len(solved)} / {len(tasks)} ({100*len(solved)/len(tasks):.1f}%)")
    print(f"   по синтезаторам: {dict(by_inv)}")
    print(f"   из них НОВЫХ (фикс-DSL не брал):   {len(new_solved)}")
    only_dsl = DSL_SOLVED - {tid for tid, _ in solved}
    union = DSL_SOLVED | {tid for tid, _ in solved}
    print(f"\n   фикс-DSL: {len(DSL_SOLVED)};  изобретение: {len(solved)};  "
          f"ОБЪЕДИНЕНИЕ: {len(union)} ({100*len(union)/len(tasks):.1f}%)")
    print(f"   потолок сдвинут на: +{len(union) - len(DSL_SOLVED)} задач\n")

    if new_solved:
        print("   НОВЫЕ задачи (id → изобретённый примитив):")
        for tid, name in new_solved:
            print(f"      {tid}: {name}")

    print("\n── Что это значит ──")
    print("   Это НЕ перебор готового: для каждой задачи система построила конкретную операцию")
    print("   из её данных и проверила на всех парах. Новые решённые — те, что фиксированный")
    print("   язык взять не мог в принципе. Если объединение > 26 — потолок реально сдвинут.")


if __name__ == "__main__":
    main()
