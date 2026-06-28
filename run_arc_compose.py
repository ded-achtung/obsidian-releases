#!/usr/bin/env python3
"""Эксперимент: система САМА компонует схему из атомов (не написана руками).

Движок synthesize ищет композицию типизированных атомов + хвост с параметрами из данных.
Меряем на реальном ARC: сколько решает композиционный синтез и сколько НОВЫХ сверх
ручных синтезаторов (объединение фикс-DSL ∪ invent = 53 на train).

Запуск: python run_arc_compose.py [--split train|eval] [--max-prefix 2] [--seconds 600]
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np
import arckit

from thinking_system.reasoning.grids import to_grid
from thinking_system.reasoning.synthesize import synthesize

# объединение фикс-DSL (26) ∪ invent (27 новых) = 53 решённых на train (из ARC_RESULTS.md)
PRIOR_UNION_TRAIN = 53


def grids(task):
    g = lambda a: to_grid(np.asarray(a).tolist())
    return [(g(i), g(o)) for i, o in task.train], [(g(i), g(o)) for i, o in task.test]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "eval"], default="train")
    p.add_argument("--max-prefix", type=int, default=2)
    p.add_argument("--seconds", type=float, default=600.0)
    args = p.parse_args()
    train_set, eval_set = arckit.load_data()
    tasks = list(train_set if args.split == "train" else eval_set)

    print(f"▶ Композиционный синтез (система строит схему сама), ARC {args.split} "
          f"({len(tasks)} задач, префикс ≤{args.max_prefix})\n", flush=True)
    start = time.time()
    solved, by_shape = [], Counter()
    composed = []                                            # схемы длиной ≥2 (реально скомпонованы)
    attempted = 0
    for task in tasks:
        if time.time() - start > args.seconds:
            break
        attempted += 1
        train, test = grids(task)
        try:
            label, fn = synthesize(train, max_prefix=args.max_prefix)
        except Exception:  # noqa: BLE001
            label, fn = None, None
        if fn is None:
            continue
        try:
            ok = all(fn(ti) == to for ti, to in test)
        except Exception:  # noqa: BLE001
            ok = False
        if ok:
            solved.append((task.id, label))
            by_shape[label.count("▸")] += 1
            if "▸" in label:
                composed.append((task.id, label))

    elapsed = time.time() - start
    print(f"── Результат за {elapsed:.0f}с (попробовано {attempted}) ──")
    print(f"   решено КОМПОЗИЦИЕЙ (точный тест): {len(solved)} / {len(tasks)} ({100*len(solved)/max(attempted,1):.1f}%)")
    print(f"   длина схемы (число ▸): {dict(by_shape)}")
    print(f"   из них РЕАЛЬНО скомпонованных (длина ≥2): {len(composed)}")
    if args.split == "train":
        print(f"   против ручных схем (фикс-DSL ∪ invent = {PRIOR_UNION_TRAIN}): "
              f"композиция нашла {len(solved)}; прирост от композиции = {len(solved) - PRIOR_UNION_TRAIN:+d}")

    if composed:
        print("\n   СХЕМЫ, скомпонованные системой (id → программа):")
        for tid, label in composed[:30]:
            print(f"      {tid}: {label}")

    print("\n── Честно ──")
    print("   Схему здесь строит ПОИСК по атомам, а не автор: новые цепочки рождаются")
    print("   композицией. Если на eval всё ещё ~0 — значит и композиция атомов не достаёт")
    print("   до нужных понятий; следующий уровень — порождать сами атомы из восприятия.")


if __name__ == "__main__":
    main()
