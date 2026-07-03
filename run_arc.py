#!/usr/bin/env python3
"""ЧЕСТНЫЙ замер на реальном ARC-AGI-1 — воспроизводимый из репозитория.

Протокол (как в самом ARC):
  1) программа ищется поиском (best_first_induce) ТОЛЬКО по train-парам задачи;
  2) задача засчитана, только если программа даёт точное совпадение на ВСЕХ
     СКРЫТЫХ test-входах (программа их при поиске не видела);
  3) рост библиотеки: комбо абстрагируются из решений на сплите training,
     а выигрыш меряется на ОТЛОЖЕННОМ сплите evaluation — задачи роста
     и задачи замера не пересекаются.

Это замер честности «как есть»: язык — общий перцептивный seed (23 примитива),
никакой подгонки под конкретные задачи. Ожидаемо решаются единицы процентов —
ценность в том, что число НАСТОЯЩЕЕ и его видно, как двигать.

Запуск: python run_arc.py                     # оба сплита + рост библиотеки
        python run_arc.py --split training    # один сплит
        python run_arc.py --limit 50          # первые N задач (быстрый прогон)
"""

from __future__ import annotations

import argparse
from collections import Counter

from thinking_system.reasoning.arc_data import ArcTask, load_arc
from thinking_system.reasoning.grid_seed import guarded_grid_seed
from thinking_system.reasoning.induction import Primitive, Program
from thinking_system.reasoning.library_learning import LibraryLearner
from thinking_system.reasoning.search_prior import best_first_induce


def evaluate(tasks: list[ArcTask], primitives: list[Primitive], *,
             max_depth: int, budget: int) -> dict:
    """Прогнать протокол по задачам; вернуть решения и статистику поиска."""
    found: dict[str, Program] = {}       # программа согласована со всеми train-парами
    correct: dict[str, Program] = {}     # …и точна на всех скрытых test-парах
    checked_total = 0
    for t in tasks:
        prog, n = best_first_induce(list(t.train), primitives, None,
                                    max_depth=max_depth, budget=budget)
        checked_total += n
        if prog is None:
            continue
        found[t.task_id] = prog
        try:
            ok = all(prog(i) == o for i, o in t.test)
        except Exception:  # noqa: BLE001 — примитив упал на тест-входе
            ok = False
        if ok:
            correct[t.task_id] = prog
    return {"found": found, "correct": correct, "checked": checked_total}


def report(name: str, res: dict, n_tasks: int) -> None:
    found, correct = res["found"], res["correct"]
    print(f"   {name}: задач {n_tasks}, программа по train-парам найдена: {len(found)}, "
          f"верна на СКРЫТЫХ test: {len(correct)} ({100 * len(correct) / n_tasks:.1f}%), "
          f"проверено программ: {res['checked']}")
    depth = Counter(p.length for p in correct.values())
    if depth:
        print(f"      глубина верных решений: {dict(sorted(depth.items()))}")
    for tid, prog in sorted(correct.items()):
        print(f"      {tid}: «{prog}»")
    lost = sorted(set(found) - set(correct))
    if lost:
        print(f"      согласованы с train, но НЕ прошли скрытый test ({len(lost)}): "
              + ", ".join(f"{tid} «{found[tid]}»" for tid in lost))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["training", "evaluation", "both"], default="both")
    ap.add_argument("--depth", type=int, default=2, help="макс. глубина композиции")
    ap.add_argument("--budget", type=int, default=4000, help="лимит проверенных программ на задачу")
    ap.add_argument("--limit", type=int, default=None, help="только первые N задач сплита")
    ap.add_argument("--data-dir", default=None, help="каталог с JSON ARC (иначе arckit)")
    ap.add_argument("--no-grow", action="store_true", help="без стадии роста библиотеки")
    args = ap.parse_args()

    seed = guarded_grid_seed()
    splits = ["training", "evaluation"] if args.split == "both" else [args.split]
    tasks = {s: load_arc(s, args.data_dir)[: args.limit] for s in splits}
    print(f"▶ Реальный ARC-AGI-1, протокол train-пары → скрытый test; "
          f"seed {len(seed)} примитивов, глубина {args.depth}, бюджет {args.budget}\n")

    print("── Базовый замер (seed-библиотека) ──")
    base = {s: evaluate(tasks[s], seed, max_depth=args.depth, budget=args.budget) for s in splits}
    for s in splits:
        report(s, base[s], len(tasks[s]))

    if args.no_grow or "training" not in base or "evaluation" not in base:
        return

    # рост: абстракции ТОЛЬКО из решений training-сплита (верных на скрытых test);
    # замер выигрыша — на непересекающемся сплите evaluation
    learner = LibraryLearner(seed)
    added = learner.grow_from_solutions(list(base["training"]["correct"].values()),
                                        top=5, min_count=2)
    print(f"\n── Рост библиотеки из решений training ({len(base['training']['correct'])} программ) ──")
    print(f"   абстрагированы повторяющиеся комбо: {added if added else 'нет повторов'}")
    if not added:
        return

    grown = evaluate(tasks["evaluation"], learner.lib.prims,
                     max_depth=args.depth, budget=args.budget)
    print("\n── Замер на ОТЛОЖЕННОМ сплите evaluation: seed vs выросшая библиотека ──")
    report("evaluation (seed)  ", base["evaluation"], len(tasks["evaluation"]))
    report("evaluation (grown) ", grown, len(tasks["evaluation"]))
    b, g = base["evaluation"], grown
    print("\n── Итог ──")
    print(f"   охват:         {len(b['correct'])} → {len(g['correct'])} верных задач")
    print(f"   стоимость:     {b['checked']} → {g['checked']} проверенных программ")
    print("   Рост библиотеки из чужого сплита честно показывает, даёт ли опыт")
    print("   решений выигрыш на невиданных задачах — в охвате и/или в поиске.")


if __name__ == "__main__":
    main()
