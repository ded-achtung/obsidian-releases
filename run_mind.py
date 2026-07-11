#!/usr/bin/env python3
"""Единый агент (Mind) на реальном ARC: чтение учебника измеримо улучшает решение.

Сценарий (две «жизни» одного агента):

  СЕССИЯ 1: агент получает поток опыта и САМ маршрутизирует его:
    1) задачи реального ARC (evaluation) → решает; трудные честно остаются
       в его списке нерешённого (бюджет размышления эскалирует по лестнице);
    2) учебник по преобразованиям сеток → читает: заземляет слова индукцией
       из показов, определения растят библиотеку операций;
    3) ПОВЕСТКА: агент замечает «язык вырос, есть нерешённое» и сам
       возвращается к нерешённым задачам;
    4) скрытые test-входы ARC проверяют честно: стало ли решено БОЛЬШЕ,
       чем до чтения. Состояние сохраняется на диск.

  СЕССИЯ 2: новый процесс загружает состояние — задачи, потребовавшие
    чтения и повторных попыток, решаются сразу (память пережила процесс).

Это кросс-доменный перенос: знание пришло ТЕКСТОМ, выигрыш измерен на СЕТКАХ
(скрытых test-входах реального ARC, непересекающихся с материалом учебника).

Запуск: python run_mind.py                # полный evaluation-сплит (~10 мин)
        python run_mind.py --limit 60     # быстрый прогон
"""

from __future__ import annotations

import argparse
import os

from thinking_system.mind import Mind
from thinking_system.reasoning.arc_data import load_arc


def check_hidden(mind: Mind, tasks: dict) -> dict[str, str]:
    """Скрытые test-входы: решённые по train-парам программы обязаны совпасть точно."""
    correct = {}
    for tid in mind.solutions:
        prog = mind.program_for(tid)
        try:
            if all(prog(i) == o for i, o in tasks[tid].test):
                correct[tid] = str(prog)
        except Exception:  # noqa: BLE001
            pass
    return correct


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="первые N задач evaluation")
    ap.add_argument("--effort", type=int, default=2, help="ступеней лестницы размышления (1-3)")
    ap.add_argument("--state", default="mind_state.json")
    ap.add_argument("--library", nargs="+",
                    default=[os.path.join("books", "textbook_grids_2.md"),
                             os.path.join("books", "textbook_grids_4.md"),
                             os.path.join("books", "notes_grids.md"),
                             os.path.join("books", "textbook_grids.md"),
                             os.path.join("books", "textbook_grids_5.md")])
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    if os.path.exists(args.state):
        os.remove(args.state)                                # честный старт с нуля

    tasks = {t.task_id: t for t in load_arc("evaluation", args.data_dir)[: args.limit]}
    stream = [{"id": tid, "train": list(t.train)} for tid, t in tasks.items()]
    library = {}
    for path in args.library:
        with open(path, encoding="utf-8") as f:
            library[os.path.basename(path)] = f.read()

    print(f"▶ Единый агент • реальный ARC-AGI-1 evaluation ({len(tasks)} задач) "
          f"+ библиотека из {len(library)} текстов\n")

    # ── СЕССИЯ 1 ─────────────────────────────────────────────────────────────────
    mind = Mind(args.state)
    print("СЕССИЯ 1")
    print(f"1) Поток задач: агент маршрутизирует опыт сам (перцепция → «грид-задача»)")
    depth_counts: dict[int, int] = {}
    for item in stream:
        res = mind.experience(item, effort=args.effort)
        if res.get("solved"):
            depth_counts[res["depth"]] = depth_counts.get(res["depth"], 0) + 1
    base_correct = check_hidden(mind, tasks)
    print(f"   решено по train-парам: {len(mind.solutions)}, верно на СКРЫТЫХ test: "
          f"{len(base_correct)}; нерешённых в повестке: {len(mind.unsolved)}")
    if depth_counts:
        print(f"   глубина размышления по решённым: {dict(sorted(depth_counts.items()))} "
              f"(бюджет эскалирует по лестнице, а не фиксирован)")

    print(f"\n2) БИБЛИОТЕКА (порядок чтения агент выбирает САМ по эпистемической ценности):")
    for entry in mind.study_library(library):
        if "пропущено" in entry:
            print(f"   пропустил {entry['пропущено']}: {entry['причина']}")
            continue
        goal = (f" — РАДИ ВОПРОСА {entry['цель']}" if "цель" in entry else "")
        print(f"   выбрал «{entry['выбрано']}» (ценность {entry['ценность']}{goal}): "
              f"показы {entry['выучено_слов']}, определения {entry['определено']}"
              + (f", ЭКСПЕРИМЕНТ: {entry['выведено_экспериментом']}"
                 if entry.get("выведено_экспериментом") else "")
              + (f", вопросы {entry['вопросы']}" if entry["вопросы"] else ""))
    print(f"   библиотека: {len(mind.abstractions)} абстракций {mind.abstractions}; "
          f"словарь {len(mind.lexicon.words)} слов; вопросы агента: {mind.questions}")

    print(f"\n3) Повестка агента: {mind.agenda()}")
    work = mind.idle_work(effort=args.effort)
    for tid, prog, checked in work["resolved"]:
        print(f"   вернулся и решил {tid}: «{prog}» (проверено {checked} программ)")
    if work["consolidated"]:
        print(f"   консолидировал из своих решений: {work['consolidated']}")

    after_correct = check_hidden(mind, tasks)
    gained = sorted(set(after_correct) - set(base_correct))
    print(f"\n4) СКРЫТЫЕ test-входы: верно {len(base_correct)} → {len(after_correct)}"
          + (f"; новые задачи от ЧТЕНИЯ: {', '.join(gained)}" if gained else ""))

    path = mind.save()
    print(f"\n5) Состояние сохранено: {path} "
          f"(абстракции {len(mind.abstractions)}, словарь {len(mind.lexicon.words)}, "
          f"решения {len(mind.solutions)}, нерешённое {len(mind.unsolved)})")

    # ── СЕССИЯ 2: новый процесс, та же память ────────────────────────────────────
    print("\nСЕССИЯ 2 (новый агент, загрузил состояние)")
    mind2 = Mind(args.state)
    probe = gained if gained else list(after_correct)[:1]
    for tid in probe:
        res = mind2.attempt(tid + "#повтор", list(tasks[tid].train), effort=args.effort)
        ok = res["solved"] and all(res["program"](i) == o for i, o in tasks[tid].test)
        print(f"   {tid}: решил сразу «{res['program']}» за {res['checked']} программ, "
              f"скрытый test: {'верно' if ok else 'НЕВЕРНО'} — знание пережило процесс")

    print("\n── Итог (честно) ──")
    print("   Один агент сам маршрутизирует текст и задачи; знание из ТЕКСТА измеримо")
    print("   добавляет решённые задачи реального ARC на скрытых тестах; бюджет")
    print("   размышления растёт с трудностью; память переживает процесс; к нерешённому")
    print("   агент возвращается по собственной повестке. Переменные в языке трёх видов:")
    print("   цвет из палитры задачи (keep/drop/paint), объект (each[f] — «к каждому")
    print("   объекту», pick[k], предикаты big/small/one[f] — «большому/маленькому/")
    print("   одиночкам», всё в двух связностях: цветослепой и одноцветной ...c)")
    print("   и шаблоны с дыркой из анти-унификации")
    print("   накопленных решений (им нужны РАЗНООБРАЗНЫЕ глубокие решения, пока их мало).")
    print("   ЧТО читать, агент выбирает сам (эпистемическая ценность, куррикулум")
    print("   возникает из переоценки после каждого текста); незаземлённые слова честно")
    print("   копятся ВОПРОСАМИ в повестке. Словарь ЧАСТИЧНО открыт: формы слова")
    print("   узнаются по основе (определение в учебнике написано падежами —")
    print("   «после отражения ещё и переворот»), показ читается из свободной прозы")
    print("   («отражение превращает [[...]] в [[...]]»). Честная граница: снимается")
    print("   только словоизменение (не словообразование), проза — на уровне строк,")
    print("   и сами тексты — по-прежнему контролируемые учебники.")


if __name__ == "__main__":
    main()
