"""Абстракции ИЗ ЧТЕНИЯ: учебник определяет новые операции через известные.

Раньше система растила язык из СВОИХ решений (library_learning). Здесь язык растёт
из МАТЕРИАЛА: учебник ОПРЕДЕЛЯЕТ новую операцию через уже выученные («X это сначала
A потом B»). Система разбирает определение, добавляет композицию в словарь как новое
слово и пользуется им в задачах. Абстракция приходит из текста, а не только из
самостоятельного решения.
"""

from __future__ import annotations

from collections import defaultdict

from thinking_system.language.understanding import GroundedLexicon, tokenize
from thinking_system.language.reader import parse_demonstration

_SEQ = {"сначала", "потом", "затем", "после"}


def parse_definition(line: str, lex: GroundedLexicon):
    """«<новое> это сначала <A> потом <B> …» → (новое_слово, [операции по порядку])."""
    toks = tokenize(line)
    if "это" not in toks:
        return None
    i = toks.index("это")
    left, right = toks[:i], toks[i + 1:]
    ops = [lex.normalize(w) for w in right if lex.normalize(w) in lex.words]
    has_seq = any(w in _SEQ for w in right)
    new = [lex.normalize(w) for w in left if lex.normalize(w) not in lex.words and lex.normalize(w) not in lex.stop]
    if len(new) == 1 and len(ops) >= 2 and has_seq:        # определение: 1 новое слово = ≥2 известных по порядку
        return new[0], ops
    return None


class DefinitionReader:
    """Учит базовые операции из примеров и НОВЫЕ операции из определений в тексте."""

    def __init__(self, *, normalize=None) -> None:
        self.lex = GroundedLexicon(normalize=normalize)
        self.definitions: dict[str, list[str]] = {}
        self.exercises: list[str] = []

    def study(self, text: str) -> dict:
        demos: dict[str, list] = defaultdict(list)
        defs: list[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            demo = parse_demonstration(line)
            if demo:
                w, inp, out = demo
                demos[w].append((inp, out))
            elif line.lower().startswith("задача"):
                self.exercises.append(line.split(":", 1)[1].strip())
            else:
                defs.append(line)
        for w, ex in demos.items():                        # 1) выучить базовые операции
            self.lex.learn(w, ex)
        for line in defs:                                  # 2) выучить определения через известные
            parsed = parse_definition(line, self.lex)
            if parsed:
                word, ops = parsed
                if self.lex.define(word, ops):
                    self.definitions[word] = ops
        return {"операции": len(self.lex.words), "определено": len(self.definitions), "задачи": len(self.exercises)}

    def solve(self, task: str) -> dict:
        return self.lex.solve(task)

    def solve_exercises(self) -> list[dict]:
        return [{"задача": t, **self.solve(t)} for t in self.exercises]
