#!/usr/bin/env python3
"""Абстракции ИЗ ЧТЕНИЯ: учебник определяет новые операции, система их усваивает.

Язык растёт не только из своих решений (library_learning), но и из МАТЕРИАЛА:
учебник ОПРЕДЕЛЯЕТ новую операцию через известные («X это сначала A потом B»).
Система разбирает определение, добавляет композицию в словарь и решает ею задачи.

Запуск: python run_definitions.py
"""

from __future__ import annotations

from thinking_system.language.definitions import DefinitionReader
from thinking_system.text.ingest import load_book


def main():
    print("▶ Рост языка ИЗ ЧТЕНИЯ: учебник определяет новые операции\n")
    r = DefinitionReader()
    stats = r.study(load_book("books/textbook_definitions.md"))

    print(f"1) ПРОШТУДИРОВАЛ: базовых операций {stats['операции'] - stats['определено']}, "
          f"определено из текста {stats['определено']}, задач {stats['задачи']}")
    print("   выучено что делают (из примеров):")
    for w, ops in [(w, p) for w, p in r.lex.words.items() if w not in r.definitions]:
        print(f"     {w} = {ops}")
    print("   ОПРЕДЕЛЕНО ИЗ ТЕКСТА (новые операции через известные):")
    for w, ops in r.definitions.items():
        print(f"     {w} = {' ▸ '.join(ops)}   (= {r.lex.words[w]})")

    print("\n2) РЕШАЕТ задачи операциями, ОПРЕДЕЛЁННЫМИ В ТЕКСТЕ:")
    for res in r.solve_exercises():
        print(f"   «{res['задача']:<20}» → {res['answer']}   (через «{res['reasoning']}»)")

    print("\n── Что это значит ──")
    print("   Система наращивает язык ИЗ МАТЕРИАЛА: прочитала определение новой операции")
    print("   через известные и стала ею пользоваться — абстракция из ТЕКСТА, а не только из")
    print("   собственных решений. Вместе с library_learning это два источника роста языка:")
    print("   из опыта (свои решения) и из чтения (определения в учебнике).")


if __name__ == "__main__":
    main()
