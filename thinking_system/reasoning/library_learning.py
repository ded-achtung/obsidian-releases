"""Самонаращивающаяся библиотека: расти язык из решённых задач, а не кодировать руками.

Урок ARC: кодировать DSL под каждый бенчмарк — тупик. Путь к «любой задаче» — система
сама РАСТИТ свой язык. Цикл (как в DreamCoder):

  • wake  — решить задачи поиском программ над ТЕКУЩЕЙ библиотекой;
  • sleep — найти переиспользуемые куски в решениях и добавить их как новые
            примитивы (сжатие/абстракция).

С маленького ОБЩЕГО набора система над потоком задач (любых доменов) наращивает
библиотеку и начинает брать всё более сложные задачи при той же глубине поиска —
без ручного кодирования под домен. Абстракция компонует существующее; для атомарно
новых операций есть синтез (dsl_growth) — вместе это путь к общности.
"""

from __future__ import annotations

import json
from collections import Counter

from thinking_system.reasoning.induction import Program, Library, default_primitives


class LibraryLearner:
    """Решает задачи и наращивает библиотеку абстракциями из найденных решений."""

    def __init__(self, seed=None) -> None:
        self.lib = Library(seed if seed is not None else default_primitives())
        self.history: list[dict] = []

    def wake(self, tasks: list[list[tuple]], *, max_depth: int = 2) -> dict:
        """Решить каждую задачу (примеры вход→выход) поиском над текущей библиотекой."""
        solutions: dict[int, Program] = {}
        for i, examples in enumerate(tasks):
            prog = self.lib.induce(examples, max_depth=max_depth)
            if prog is not None:
                solutions[i] = prog
        return solutions

    def _by_name(self, name: str):
        return next((p for p in self.lib.prims if p.name == name), None)

    def sleep(self, solutions: dict[int, Program], *, top: int = 1, min_count: int = 2) -> list[str]:
        """Абстрагировать самые частые/сжимающие подпоследовательности решений в примитивы.

        Комбо с шагами, которых нет в текущей библиотеке (решение чужого решателя над
        другим словарём), пропускаются молча — абстрагируется только то, что можно
        собрать из известных примитивов.
        """
        counts: Counter = Counter()
        for prog in solutions.values():
            names = [s.name for s in prog.steps]
            for length in range(2, len(names) + 1):
                for i in range(len(names) - length + 1):
                    counts[tuple(names[i:i + length])] += 1
        # сжатие ≈ выигрыш = (частота−1)·(длина−1); берём непокрытые именами уже существующих абстракций
        existing = {p.name for p in self.lib.prims}
        cands = [(seq, c) for seq, c in counts.items() if c >= min_count and "∘".join(seq) not in existing]
        added: list[str] = []
        for seq, c in sorted(cands, key=lambda kv: -((kv[1] - 1) * (len(kv[0]) - 1))):
            if len(added) >= top:
                break
            steps = [self._by_name(n) for n in seq]
            if any(s is None for s in steps):                # шаг вне библиотеки
                continue
            name = "∘".join(seq)
            self.lib.add_abstraction(name, Program(steps))
            added.append(name)
        return added

    # ── сериализация: рост переживает процесс ──────────────────────────────────
    # Абстракция = последовательность имён шагов, поэтому библиотека восстановима
    # поверх того же seed. Сохраняются только выученные комбо, не seed-примитивы.

    def save(self, path: str) -> None:
        """Сохранить выученные абстракции (имена-цепочки) в JSON."""
        combos = [name.split("∘") for name in self.lib.abstractions]
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"abstractions": combos}, f, ensure_ascii=False, indent=1)

    def load(self, path: str) -> list[str]:
        """Восстановить абстракции поверх текущего seed; вернуть добавленные имена.

        Комбо, чьи шаги неизвестны текущей библиотеке, пропускаются (это честное
        поведение: библиотека растёт только из того, что умеет исполнить).
        """
        with open(path, encoding="utf-8") as f:
            combos = json.load(f)["abstractions"]
        existing = {p.name for p in self.lib.prims}
        added: list[str] = []
        for seq in combos:
            name = "∘".join(seq)
            steps = [self._by_name(n) for n in seq]
            if name in existing or any(s is None for s in steps):
                continue
            self.lib.add_abstraction(name, Program(steps))
            added.append(name)
        return added

    def grow_from_solutions(self, programs, *, top: int = 2, min_count: int = 2) -> list[str]:
        """Вырастить библиотеку из РЕШЕНИЙ, найденных ЛЮБЫМ решателем (не только своим wake).

        Принимает готовые Program — например, найденные best_first_induce на реальном ARC, —
        и абстрагирует их частые/уникальные подпоследовательности в новые примитивы. Так
        библиотека растёт из НАСТОЯЩИХ решений системы на реальных задачах, а не только из
        того, что решил её собственный поиск в wake. При min_count=1 запоминает даже
        одиночные найденные композиции как «выученные ходы» (память на решённые подзадачи).
        """
        sols = {i: p for i, p in enumerate(programs) if p is not None and getattr(p, "steps", None)}
        return self.sleep(sols, top=top, min_count=min_count)

    def learn(self, tasks: list[list[tuple]], *, rounds: int = 3, max_depth: int = 2, abstractions_per_round: int = 1) -> list[dict]:
        """Чередовать wake/sleep по потоку задач: библиотека растёт, решается больше."""
        for r in range(rounds):
            sols = self.wake(tasks, max_depth=max_depth)
            self.history.append({"round": r, "solved": len(sols), "library": len(self.lib.prims),
                                 "abstractions": list(self.lib.abstractions)})
            self.sleep(sols, top=abstractions_per_round)
        return self.history

    def solve(self, examples: list[tuple], *, max_depth: int = 2) -> Program | None:
        return self.lib.induce(examples, max_depth=max_depth)
