"""Образование понятий снизу-вверх: открыть переиспользуемые абстракции по MDL.

Прежняя абстракция (library_learning) брала частые подпоследовательности по эвристике
(частота×длина). Здесь — ПРИНЦИПИАЛЬНЫЙ критерий: минимум длины описания (MDL, ядро
DreamCoder). Система смотрит на СВОИ решения (программы над базовым языком) и сама
находит «понятия» — подпрограммы, которые СЖИМАЮТ корпус в битах:

    DL = (токенов в переписанном корпусе)·log2(|словарь|) + (длина определений)·log2(|база|)

Добавление абстракции расширяет словарь (токен дороже), но сокращает число токенов
(повтор → один символ). MDL честно балансирует это и САМ решает, что абстрагировать и
когда остановиться — без порога «частота≥2» и без подсказки, какое понятие нужно.

Честно: «понятие» здесь = переиспользуемая сжимающая подпрограмма над уже имеющимися
операциями (формальная абстракция из опыта), а не понятие из сырого восприятия и не
человеческий концепт. Но открывается оно автономно — из данных, по объективному критерию.
"""

from __future__ import annotations

import math
from collections import Counter

from thinking_system.reasoning.induction import Library, Primitive, Program


def _rewrite(prog_names: list[str], abstractions: dict[str, tuple[str, ...]]) -> list[str]:
    """Переписать программу, жадно заменяя вхождения абстракций (длинные — первыми)."""
    if not abstractions:
        return list(prog_names)
    order = sorted(abstractions.items(), key=lambda kv: -len(kv[1]))
    out, i = [], 0
    names = list(prog_names)
    while i < len(names):
        for name, seq in order:
            n = len(seq)
            if tuple(names[i:i + n]) == seq:
                out.append(name)
                i += n
                break
        else:
            out.append(names[i])
            i += 1
    return out


def _expand(seq: tuple[str, ...], abstractions: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    """Развернуть абстракцию до базовых операций (для длины определения и исполнения)."""
    out: list[str] = []
    for n in seq:
        if n in abstractions:
            out.extend(_expand(abstractions[n], abstractions))
        else:
            out.append(n)
    return tuple(out)


def description_length(corpus: list[list[str]], abstractions: dict[str, tuple[str, ...]], n_base: int) -> float:
    """Полная длина описания корпуса в битах при данном наборе абстракций."""
    n_symbols = n_base + len(abstractions)
    per_tok = math.log2(max(n_symbols, 2))
    corpus_tokens = sum(len(_rewrite(p, abstractions)) for p in corpus)
    defs_tokens = sum(len(_expand(seq, abstractions)) for seq in abstractions.values())
    per_base = math.log2(max(n_base, 2))
    return corpus_tokens * per_tok + defs_tokens * per_base


def _candidates(corpus: list[list[str]], abstractions: dict[str, tuple[str, ...]]) -> Counter:
    """Все смежные подпоследовательности (длина≥2) переписанного корпуса и их частоты."""
    counts: Counter = Counter()
    for p in corpus:
        names = _rewrite(p, abstractions)
        for length in range(2, len(names) + 1):
            for i in range(len(names) - length + 1):
                counts[tuple(names[i:i + length])] += 1
    return counts


class MDLLearner:
    """Открывает абстракции из корпуса решений, минимизируя длину описания (биты)."""

    def __init__(self, base_prims: list[Primitive]) -> None:
        self.base = list(base_prims)
        self.by_name = {p.name: p for p in self.base}
        self.abstractions: dict[str, tuple[str, ...]] = {}   # имя → базовая (развёрнутая) последовательность
        self.steps: list[dict] = []

    def compress(self, corpus: list[list[str]], *, max_abstractions: int = 8) -> list[dict]:
        """Жадно добавлять абстракцию с наибольшим выигрышем DL, пока он положителен."""
        n_base = len(self.base)
        dl = description_length(corpus, self.abstractions, n_base)
        for _ in range(max_abstractions):
            cur = self.abstractions
            best, best_dl, best_name = None, dl, None
            for seq, c in _candidates(corpus, cur).items():
                if c < 2:
                    continue
                expanded = _expand(seq, cur)               # определение над базой
                name = "∘".join(expanded)
                if name in cur:
                    continue
                trial = {**cur, name: expanded}
                d = description_length(corpus, trial, n_base)
                if d < best_dl - 1e-9:
                    best, best_dl, best_name = expanded, d, name
            if best is None:
                break
            self.abstractions[best_name] = best
            self.steps.append({"concept": best_name, "len": len(best),
                               "dl_before": round(dl, 1), "dl_after": round(best_dl, 1),
                               "bits_saved": round(dl - best_dl, 1)})
            dl = best_dl
        return self.steps

    def to_library(self, *, max_depth_seed: list[Primitive] | None = None) -> Library:
        """Собрать Library: база + открытые абстракции как примитивы (исполнимые)."""
        lib = Library(max_depth_seed if max_depth_seed is not None else self.base)
        for name, expanded in self.abstractions.items():
            prog = Program([self.by_name[n] for n in expanded])
            lib.add_abstraction(name, prog)
        return lib

    def total_bits(self, corpus: list[list[str]], *, with_abstractions: bool) -> float:
        return description_length(corpus, self.abstractions if with_abstractions else {}, len(self.base))
