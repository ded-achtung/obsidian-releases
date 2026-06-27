#!/usr/bin/env python3
"""Автономные понятия снизу-вверх: система открывает абстракции из СВОИХ решений по MDL.

Разрыв «все примитивы заданы руками» — честный измеримый шаг: система НЕ получает
понятие в руки. Она (1) сама решает поток задач поиском над базовым языком, (2) по
объективному критерию MDL находит в своих решениях переиспользуемые подпрограммы
(сжимающие корпус в битах) и называет их, (3) с этими понятиями берёт НОВЫЕ задачи
на меньшей глубине поиска.

Бейзлайн — базовый язык без абстракций. Метрики честные: биты до/после (реальное
сжатие) и решаемость held-out задач на глубине 1 (которой раньше не было).

Запуск: python run_concepts.py
"""

from __future__ import annotations

from thinking_system.reasoning.induction import Library, default_primitives
from thinking_system.reasoning.mdl_abstraction import MDLLearner

# Поток задач (числа/списки). Решения разделяют ДВА скрытых понятия, но какие именно —
# системе не сообщается; она найдёт их сама. Каждое встречается несколько раз.
TASKS = [
    [(2, 9), (3, 16), (5, 36)],                                  # (x+1)²        = +1 ▸ square
    [(2, 18), (3, 32), (4, 50)],                                 # 2·(x+1)²      = +1 ▸ square ▸ *2
    [(1, 5), (2, 10), (3, 17)],                                  # (x+1)²+1      = +1 ▸ square ▸ +1
    [(2, 11), (3, 18), (4, 27)],                                 # (x+1)²+2      = +1 ▸ square ▸ +2
    [([1, 2, 3], 12), ([5], 10), ([2, 2], 8)],                   # 2·sum         = each*2 ▸ sum
    [([1, 2, 3], 144), ([5], 100), ([2, 2], 64)],                # (2·sum)²      = each*2 ▸ sum ▸ square
    [([1, 2, 3], 13), ([5], 11), ([2, 2], 9)],                   # 2·sum+1       = each*2 ▸ sum ▸ +1
    [([1, 2, 3], 14), ([5], 12), ([2, 2], 10)],                  # 2·sum+2       = each*2 ▸ sum ▸ +2
]

# held-out: новые задачи тех же понятий — проверяем решаемость на глубине 1
HELD_OUT = [
    ("(x+1)²", [(6, 49), (7, 64), (9, 100)]),                    # нужно понятие +1 ▸ square
    ("2·sum", [([4, 1], 10), ([10], 20), ([3, 3], 12)]),         # нужно понятие each*2 ▸ sum
]


def main():
    base = default_primitives()
    seed_lib = Library(base)

    # 1) система САМА решает поток задач поиском над базовым языком → корпус её решений
    corpus = []
    print("▶ Автономные понятия по MDL: открыть абстракции из собственных решений\n")
    print("1) РЕШЕНИЯ поиском над базовым языком (корпус, из которого растим понятия):")
    for ex in TASKS:
        prog = seed_lib.induce(ex, max_depth=3)
        assert prog is not None, "базовый язык должен решать обучающие задачи поиском"
        names = [s.name for s in prog.steps]
        corpus.append(names)
        print(f"   {str(prog)}")

    # 2) MDL открывает понятия (сжимающие подпрограммы) — без подсказки, какие нужны
    learner = MDLLearner(base)
    steps = learner.compress(corpus, max_abstractions=6)
    bits0 = learner.total_bits(corpus, with_abstractions=False)
    bits1 = learner.total_bits(corpus, with_abstractions=True)
    print(f"\n2) ОТКРЫТЫЕ ПОНЯТИЯ по MDL (минимум длины описания):")
    for s in steps:
        print(f"   «{s['concept']}»  (−{s['bits_saved']} бит: {s['dl_before']} → {s['dl_after']})")
    print(f"   длина описания корпуса: {bits0:.1f} → {bits1:.1f} бит  "
          f"(сжатие ×{bits0 / max(bits1, 1e-9):.2f})")

    # 3) с открытыми понятиями — held-out задачи берутся на глубине 1 (раньше None)
    grown = learner.to_library()
    print("\n3) HELD-OUT задачи на глубине 1 (новые числа/списки тех же понятий):")
    print(f"   {'задача':<14}{'базовый seed, d=1':<22}{'после понятий, d=1':<22}")
    base_solved = grown_solved = 0
    for label, ex in HELD_OUT:
        before = seed_lib.induce(ex, max_depth=1)
        after = grown.induce(ex, max_depth=1)
        base_solved += before is not None
        grown_solved += after is not None
        print(f"   {label:<14}{str(before):<22}{('«' + str(after) + '»'):<22}")
    print(f"   решено на d=1: базовый {base_solved}/{len(HELD_OUT)}  →  с понятиями {grown_solved}/{len(HELD_OUT)}")

    print("\n── Что это значит ──")
    print("   Понятия не заданы руками — система открыла их в СВОИХ решениях по объективному")
    print("   критерию (минимум бит) и тем сжала корпус; с ними новые задачи решаются меньшим")
    print("   поиском. Честно: «понятие» здесь = сжимающая подпрограмма над базовыми операциями")
    print("   (формальная абстракция из опыта), не концепт из сырого восприятия.")


if __name__ == "__main__":
    main()
