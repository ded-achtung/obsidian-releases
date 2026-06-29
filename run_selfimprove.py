#!/usr/bin/env python3
"""Самообучение: провалы порождают новые схемы, репертуар растёт, решается больше.

Цикл на реальном ARC: (1) решить сидовым репертуаром (плоский invent); (2) из ПРОВАЛОВ
открыть переиспользуемые схемы (префикс ▸ хвост, объясняющие ≥2 задачи); (3) нарастить
репертуар и решить ещё. Меряем прирост и проверяем, что схемы открыты системой, не нами.

Запуск: python run_selfimprove.py [--split train|eval]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import arckit

from thinking_system.reasoning.grids import to_grid
from thinking_system.agent.self_improve import SelfImprover


def grids(t):
    g = lambda a: to_grid(np.asarray(a).tolist())
    return [(g(i), g(o)) for i, o in t.train], [(g(i), g(o)) for i, o in t.test]


def solved_on_test(fn, test):
    try:
        return all(fn(ti) == to for ti, to in test)
    except Exception:  # noqa: BLE001
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "eval"], default="train")
    args = p.parse_args()
    tr, ev = arckit.load_data()
    tasks = [(t.id, *grids(t)) for t in (tr if args.split == "train" else ev)]
    imp = SelfImprover()

    print(f"▶ Самообучение из провалов на ARC {args.split} ({len(tasks)} задач)\n", flush=True)
    t0 = time.time()

    # 1) сид: плоский репертуар
    seed_solved, failed = set(), []
    for tid, train, test in tasks:
        r = imp._flat_solves(train)
        if r is not None and solved_on_test(r[1], test):
            seed_solved.add(tid)
        else:
            failed.append((train, test, tid))
    print(f"1) СИД (плоский invent): решено {len(seed_solved)} на тесте; провалов {len(failed)}")

    # 2) SLEEP: открыть схемы из провалов (≥2 переиспользования на train)
    disc = imp.discover([(tr_, te_) for tr_, te_, _ in failed], min_reuse=2)
    print(f"\n2) ОТКРЫТО ИЗ ПРОВАЛОВ (схемы, объясняющие ≥2 задачи — система их КОПИТ):")
    print(f"   кандидаты (префикс → сколько провалов объясняет на train): {disc['candidate_counts']}")
    print(f"   ОСТАВЛЕНО в репертуар: {disc['kept']}")

    # 3) решить провалы наросшим репертуаром; считать НОВЫЕ на тесте
    new_solved = []
    for train, test, tid in failed:
        for prefix in imp.schemas:
            r = imp._solve_prefix(prefix, train)
            if r is not None and solved_on_test(r[1], test):
                new_solved.append((tid, prefix[0][0] + " ▸ " + r[0]))
                break
    grand = len(seed_solved) + len(new_solved)
    print(f"\n3) НАРОСШИЙ РЕПЕРТУАР: +{len(new_solved)} новых задач на тесте")
    print(f"   итог: сид {len(seed_solved)} → после самообучения {grand} "
          f"({100*grand/len(tasks):.1f}%);  открытых схем: {len(imp.schemas)}")
    if new_solved:
        print("   новые решения (id → схема, ОТКРЫТАЯ системой):")
        for tid, lbl in new_solved[:20]:
            print(f"      {tid}: {lbl}")

    print(f"\n── Что это значит ({time.time()-t0:.0f}с) ──")
    print("   Петля замкнута: провал → генерация схем-кандидатов → отбор переиспользуемых →")
    print("   рост репертуара → больше решений. Схемы система открыла САМА из своих провалов")
    print("   (по критерию «объясняет ≥2 задачи»), а не мы их вписали. Честно: пространство")
    print("   кандидатов (атомы) наше — это всё ещё рост ВНУТРИ заданного словаря, не за его")
    print("   пределы; на eval прорыва не ждём, но самонаращивание репертуара — настоящее.")


if __name__ == "__main__":
    main()
