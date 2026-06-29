#!/usr/bin/env python3
"""Честный замер на РЕАЛЬНОМ ARC (то, чего аудит не нашёл): сколько решит система.

Берём настоящие задачи ARC (пакет arckit, данные ARC-AGI). Для каждой задачи система
ИЩЕТ программу над своим грид-DSL (23 примитива: симметрии, гравитация, крупнейший
объект, bbox, фрактал, достройка симметрии…), согласованную со ВСЕМИ обучающими
парами, и применяет к тестовому входу. Метрика — точное совпадение тестового ВЫХОДА
(настоящий счёт ARC), плюс отдельно «подошло на train» (чтобы видеть переобучение DSL).

Запуск:  python run_arc.py --seconds 600          # 10 минут на training-сплите
         python run_arc.py --seconds 600 --split eval --depth 3 --budget 8000
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import arckit

from thinking_system.reasoning.grid_seed import full_grid_seed
from thinking_system.reasoning.grids import to_grid
from thinking_system.reasoning.induction import Program


def task_grids(task):
    g = lambda a: to_grid(np.asarray(a).tolist())
    train = [(g(i), g(o)) for i, o in task.train]
    test = [(g(i), g(o)) for i, o in task.test]
    return train, test


def _cells(g):
    return len(g) * (len(g[0]) if g else 0)


def induce_guarded(train, prims, *, depth, budget, max_cells=2000):
    """BFS по DSL (Оккам по длине) с защитой от взрыва размера сетки и бюджетом."""
    inputs = [i for i, _ in train]
    outputs = [o for _, o in train]
    consistent = lambda vals: all(v == o for v, o in zip(vals, outputs))
    if consistent(inputs):
        return Program([]), 0
    frontier = [([], inputs)]
    gen = 0
    for _ in range(depth):
        nxt = []
        for steps, vals in frontier:
            for p in prims:
                try:
                    nv = [p.fn(v) for v in vals]
                except Exception:  # noqa: BLE001 — несовместимый вход примитива
                    continue
                if any(_cells(g) > max_cells for g in nv):   # защита от fractal-взрыва
                    continue
                gen += 1
                if gen > budget:
                    return None, gen
                ns = steps + [p]
                if consistent(nv):
                    return Program(ns), gen
                nxt.append((ns, nv))
        frontier = nxt
    return None, gen


def try_task(task, prims, *, depth, budget):
    train, test = task_grids(task)
    prog, n = induce_guarded(train, prims, depth=depth, budget=budget)
    if prog is None:
        return None, n, False
    try:
        ok = all(prog(ti) == to for ti, to in test)        # точное совпадение тестового выхода
    except Exception:  # noqa: BLE001
        ok = False
    return prog, n, ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=600.0)   # бюджет 10 минут
    p.add_argument("--split", choices=["train", "eval"], default="train")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--budget", type=int, default=8000)       # программ на задачу
    p.add_argument("--limit", type=int, default=0)           # 0 = все
    args = p.parse_args()

    train_set, eval_set = arckit.load_data()
    tasks = list(train_set if args.split == "train" else eval_set)
    if args.limit:
        tasks = tasks[:args.limit]
    prims = full_grid_seed()

    print(f"▶ ЧЕСТНЫЙ замер на РЕАЛЬНОМ ARC ({args.split}): {len(tasks)} задач, DSL {len(prims)} примитивов", flush=True)
    print(f"  бюджет времени {args.seconds:.0f}с; фаза 1 — глубина 2; фаза 2 — глубина 3 на нерешённых\n", flush=True)

    start = time.time()
    solved_ids: dict[str, str] = {}
    overfit_ids: dict[str, str] = {}
    attempted = set()

    def sweep(depth, budget, only_unsolved):
        for task in tasks:
            if time.time() - start > args.seconds:
                return
            if task.id in solved_ids:
                continue
            if only_unsolved and task.id in attempted and task.id in overfit_ids:
                continue
            prog, _, ok = try_task(task, prims, depth=depth, budget=budget)
            attempted.add(task.id)
            if prog is not None:
                if ok:
                    solved_ids[task.id] = str(prog) or "id"
                    overfit_ids.pop(task.id, None)
                    print(f"   ✓ решено: {task.id}: {str(prog) or 'id'}", flush=True)
                else:
                    overfit_ids[task.id] = str(prog) or "id"

    sweep(depth=2, budget=4000, only_unsolved=False)        # фаза 1: быстрый проход глубины 2
    phase1 = len(solved_ids)
    print(f"\n   [фаза 1 (глубина 2) за {time.time()-start:.0f}с: решено {phase1}, попробовано {len(attempted)}]\n", flush=True)
    sweep(depth=3, budget=args.budget, only_unsolved=True)  # фаза 2: глубина 3 на оставшихся

    elapsed = time.time() - start
    solved = len(solved_ids)
    train_fit = solved + len(overfit_ids)
    print(f"\n── Результат за {elapsed:.0f}с ──")
    print(f"   попробовано задач:               {len(attempted)} из {len(tasks)}")
    print(f"   программа подошла на train:       {train_fit}")
    print(f"   РЕШЕНО на тесте (точный выход):   {solved}  ({100*solved/max(len(attempted),1):.1f}% попробованных, "
          f"{100*solved/len(tasks):.1f}% всего сплита)")
    print(f"   из них фаза 2 (глубина 3) добавила: {solved - phase1}")
    print(f"   переобучение DSL (train ок, тест нет): {len(overfit_ids)}")

    if solved_ids:
        print("\n   РЕШЁННЫЕ задачи (id → программа):")
        for tid, prog in sorted(solved_ids.items()):
            print(f"      {tid}: {prog}")
    print("\n── Честно ──")
    print("   Это настоящий ARC и настоящий счёт (точное совпадение тестового выхода). Низкий")
    print("   процент ожидаем: 23-примитивный DSL покрывает малую долю преобразований ARC, а")
    print("   поиск ограничен глубиной/бюджетом. Зато цифра РЕАЛЬНАЯ и воспроизводимая — в")
    print("   отличие от прежних утверждений про ARC, которые были голой прозой.")


if __name__ == "__main__":
    main()
