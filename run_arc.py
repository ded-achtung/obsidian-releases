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
             max_depth: int, budget: int, use_parametric: bool = False,
             use_objects: bool = False) -> dict:
    """Прогнать протокол по задачам; вернуть решения и статистику поиска."""
    from thinking_system.reasoning import object_param, parametric
    from thinking_system.reasoning.grid_seed import guard

    found: dict[str, Program] = {}       # программа согласована со всеми train-парами
    correct: dict[str, Program] = {}     # …и точна на всех скрытых test-парах
    checked_total = 0
    for t in tasks:
        extra = []
        if use_parametric:               # переменная-цвет связывается из палитры задачи
            extra += parametric.instantiate(list(t.train))
        if use_objects:                  # переменная-объект: each/pick + предикаты big/small/one
            extra += object_param.instantiate()
            extra += object_param.instantiate_predicates(list(t.train))
        prims = primitives + [guard(p) for p in extra]
        prog, n = best_first_induce(list(t.train), prims, None,
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


def deep_stage(tasks: dict, base: dict, seed: list, lib: list, splits: list, args,
               learner=None) -> None:
    """Глубина 3 умным поиском по НЕРЕШЁННЫМ задачам — итеративный wake/sleep.

    Раунд 1: приор — биграммы training-решений, язык — seed + абстракции из
    training-решений (lib); структуры глубины до 6 в базовых именах достижимы
    на глубине 3, новые имена без счётчиков достижимы благодаря сглаживанию
    (AUDIT, дополнение 12). Со 2-го раунда опыт = ВСЕ решения, прошедшие
    скрытые тесты (training + evaluation + deep-находки): из пула растут новые
    абстракции, приор пересчитывается — найденное в одном раунде становится
    языком следующего. Скрытый тест текущей задачи в её поиске не участвует
    никогда. Итерация честно останавливается, когда раунд не даёт новых
    верных решений."""
    from thinking_system.reasoning import object_param, parametric
    from thinking_system.reasoning.deep_search import bigram_prior, guided_induce
    from thinking_system.reasoning.grid_seed import guard

    pool = list(base.get("training", base[splits[0]])["correct"].values())
    taken = {s: set(base[s]["found"]) for s in splits}       # взятые поиском (включая переобучившиеся)
    n_correct = {s: len(base[s]["correct"]) for s in splits}
    for r in range(1, args.deep_rounds + 1):
        bigram = bigram_prior([[st.name for st in p.steps] for p in pool])
        rnd = f", раунд {r}/{args.deep_rounds}" if args.deep_rounds > 1 else ""
        print(f"\n── Умный поиск глубины 3 по нерешённым (бюджет {args.deep}/задачу; "
              f"биграммы из {len(pool)} проверенных решений + эвристика цели"
              + (f"; язык + {len(lib)} абстракций" if lib else "") + rnd + ") ──")
        round_correct: list = []
        for s in splits:
            found, correct, checked = {}, {}, 0
            for t in tasks[s]:
                if t.task_id in taken[s]:
                    continue
                extra = []
                if args.parametric:
                    extra += parametric.instantiate(list(t.train))
                if args.objects:
                    extra += object_param.instantiate() + object_param.instantiate_predicates(list(t.train))
                prog, n = guided_induce(list(t.train), seed + lib + [guard(p) for p in extra],
                                        bigram, max_depth=3, budget=args.deep)
                checked += n
                if prog is None:
                    continue
                found[t.task_id] = prog
                try:
                    if all(prog(i) == o for i, o in t.test):
                        correct[t.task_id] = prog
                except Exception:  # noqa: BLE001
                    pass
            report(f"{s} (+deep{rnd})", {"found": found, "correct": correct, "checked": checked},
                   len(tasks[s]))
            taken[s] |= set(found)
            n_correct[s] += len(correct)
            round_correct += list(correct.values())
            print(f"      итого верных на сплите с учётом глубины ≤2: {n_correct[s]} "
                  f"({100 * n_correct[s] / len(tasks[s]):.1f}%)")
        if r == args.deep_rounds:
            break
        if not round_correct:
            print("   раунд не дал новых верных решений — итерация честно остановлена")
            break
        # sleep: опыт пополняется ВСЕМИ проверенными решениями, язык растёт из пула
        if r == 1 and "evaluation" in base:
            pool += list(base["evaluation"]["correct"].values())
        pool += round_correct
        if learner is not None:
            new_abs = learner.grow_from_solutions(pool, top=5, min_count=2)
            lib = [p for p in learner.lib.prims if p.name in set(learner.lib.abstractions)]
            print(f"   sleep: пул опыта {len(pool)} решений; новые абстракции: "
                  f"{new_abs if new_abs else 'нет'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["training", "evaluation", "both"], default="both")
    ap.add_argument("--depth", type=int, default=2, help="макс. глубина композиции")
    ap.add_argument("--budget", type=int, default=4000, help="лимит проверенных программ на задачу")
    ap.add_argument("--limit", type=int, default=None, help="только первые N задач сплита")
    ap.add_argument("--data-dir", default=None, help="каталог с JSON ARC (иначе arckit)")
    ap.add_argument("--no-grow", action="store_true", help="без стадии роста библиотеки")
    ap.add_argument("--parametric", action="store_true",
                    help="+ параметрические примитивы keep/drop/paint по палитре задачи")
    ap.add_argument("--objects", action="store_true",
                    help="+ объектные переменные each[f] / pick[k]")
    ap.add_argument("--deep", type=int, default=0, metavar="BUDGET",
                    help="умный поиск глубины 3 по нерешённым (биграммы training-решений "
                         "+ эвристика цели), бюджет программ на задачу")
    ap.add_argument("--deep-rounds", type=int, default=1, metavar="N",
                    help="итеративный wake/sleep: раунды deep-поиска, между ними пул "
                         "проверенных решений пополняется и язык растёт (стоп, если "
                         "раунд не дал новых верных решений)")
    args = ap.parse_args()

    seed = guarded_grid_seed()
    splits = ["training", "evaluation"] if args.split == "both" else [args.split]
    tasks = {s: load_arc(s, args.data_dir)[: args.limit] for s in splits}
    print(f"▶ Реальный ARC-AGI-1, протокол train-пары → скрытый test; "
          f"seed {len(seed)} примитивов, глубина {args.depth}, бюджет {args.budget}"
          + (", + параметрические keep/drop/paint" if args.parametric else "")
          + (", + объектные each/pick" if args.objects else "") + "\n")

    print("── Базовый замер (seed-библиотека) ──")
    base = {s: evaluate(tasks[s], seed, max_depth=args.depth, budget=args.budget,
                        use_parametric=args.parametric, use_objects=args.objects)
            for s in splits}
    for s in splits:
        report(s, base[s], len(tasks[s]))

    # рост: абстракции ТОЛЬКО из решений training-сплита (верных на скрытых test);
    # замер выигрыша — на непересекающемся сплите evaluation. Рост идёт ДО deep,
    # чтобы язык среднего уровня участвовал в глубоком поиске.
    learner, added, lib = None, [], []
    if not args.no_grow and "training" in base:
        learner = LibraryLearner(seed)
        added = learner.grow_from_solutions(list(base["training"]["correct"].values()),
                                            top=5, min_count=2)
        lib = [p for p in learner.lib.prims if p.name in set(learner.lib.abstractions)]
        print(f"\n── Рост библиотеки из решений training ({len(base['training']['correct'])} программ) ──")
        print(f"   абстрагированы повторяющиеся комбо: {added if added else 'нет повторов'}")

    if args.deep:
        deep_stage(tasks, base, seed, lib, splits, args, learner)

    if not added or "evaluation" not in base:
        return

    grown = evaluate(tasks["evaluation"], learner.lib.prims,
                     max_depth=args.depth, budget=args.budget,
                     use_parametric=args.parametric, use_objects=args.objects)
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
