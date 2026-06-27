#!/usr/bin/env python3
"""Многошаговые задачи: спланировать последовательность правил к цели.

Система учит простые операции из примеров, а затем для задачи «получи цель из
входа» САМА ищет цепочку операций (поиск в ширину), превращающую вход в цель.
Шаги не названы — система их планирует и проверяет исполнением.

Запуск: python run_plan.py
"""

from __future__ import annotations

from thinking_system.language.study import Textbook
from thinking_system.language.planner import TaskPlanner
from thinking_system.text.ingest import load_book


def main():
    print("▶ Многошаговые задачи: планирование цепочки выученных правил к цели\n")
    tb = Textbook()
    tb.study(load_book("books/textbook_multistep.md"))
    planner = TaskPlanner(tb.lex)
    print(f"1) ВЫУЧИЛ операции из примеров: {', '.join(f'{w}={p}' for w, p in tb.lex.words.items())}")

    print("\n2) ПЛАНИРУЕТ цепочку шагов под каждую задачу (шаги НЕ названы в условии):")
    for task in tb.exercises:
        r = planner.plan(task)
        if r["solved"]:
            print(f"   «{task:<26}» → {r['answer']}   ⟵ план: {' ▸ '.join(r['plan'])}")
        else:
            print(f"   «{task:<26}» → {r['reason']}")

    print("\n3) ЧЕСТНО, когда цель недостижима выученными операциями:")
    r = planner.plan("получи 100 из [1, 2, 3]")
    print(f"   «получи 100 из [1,2,3]» → {r['reason']}")

    print("\n── Что это значит ──")
    print("   Система решает задачи, требующие НЕСКОЛЬКИХ шагов: сама находит порядок")
    print("   операций, ведущий от входа к цели (планирование над выученными правилами),")
    print("   и проверяет план исполнением. Не одно правило по слову — а композиция под цель.")


if __name__ == "__main__":
    main()
