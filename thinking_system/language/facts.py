"""Понимание утверждений: язык → факты → ВЫВОД (отвечает дедукцией, а не генерацией).

Система разбирает утверждения на естественном (контролируемом) языке в факты-
отношения, строит их ВЫВОДНЫЕ следствия (транзитивность, наследование) и отвечает
на вопросы ДЕДУКЦИЕЙ — в том числе на то, что прямо не говорилось. Это понимание
как построение инференциальной структуры: «Сократ смертен» не сказано, но СЛЕДУЕТ.

Контролируемая грамматика (узко и честно, без морфологии):
  • «A это B»             → is_a(A, B)            (членство/подкласс)
  • «все/каждый B P»      → свойство(B, P)        (универсальное свойство класса)
  • «A это B?»            → запрос is_a(A, B)     (через транзитивность)
  • «A P?»                → запрос свойство(A, P) (через наследование)
"""

from __future__ import annotations

from thinking_system.reasoning.relational import KnowledgeBase
from thinking_system.language.understanding import tokenize

_UNIV = {"все", "всё", "каждый", "каждое", "каждая", "любой"}


class FactReader:
    """Читает утверждения в базу знаний и отвечает на вопросы выводом."""

    def __init__(self) -> None:
        self.kb = KnowledgeBase()
        self.kb.transitive("is_a")           # подкласс транзитивен
        self.kb.inherits("is_a")             # свойства класса наследуются видом

    def tell(self, statement: str) -> tuple[str, str, str]:
        """Разобрать утверждение и добавить факт. Вернуть добавленную тройку.

        Бросает ValueError на пустом/неполном утверждении (например «A это» без B).
        """
        toks = tokenize(statement)
        if not toks:
            raise ValueError(f"пустое утверждение: {statement!r}")
        if any(t in _UNIV for t in toks):
            rest = [t for t in toks if t not in _UNIV]
            if len(rest) < 2:
                raise ValueError(f"неполное универсальное утверждение: {statement!r}")
            cls, prop = rest[0], rest[-1]
            self.kb.add("свойство", cls, prop)
            return ("свойство", cls, prop)
        if "это" in toks:
            i = toks.index("это")
            if not 0 < i < len(toks) - 1:
                raise ValueError(f"не распознано утверждение «A это B»: {statement!r}")
            a, b = toks[i - 1], toks[i + 1]
        else:
            a, b = toks[0], toks[-1]
        self.kb.add("is_a", a, b)
        return ("is_a", a, b)

    def parse_question(self, question: str) -> tuple[str, str, str]:
        """Разобрать вопрос в запрос (отношение, A, B)."""
        toks = tokenize(question)
        if not toks:
            return ("свойство", "", "")
        if "это" in toks:
            i = toks.index("это")
            a = toks[i - 1] if i - 1 >= 0 else toks[0]
            b = toks[i + 1] if i + 1 < len(toks) else toks[-1]
            return ("is_a", a, b)
        return ("свойство", toks[0], toks[-1])

    def ask(self, question: str) -> bool:
        """Ответить на вопрос ДЕДУКЦИЕЙ по выведенным следствиям (замкнутый мир)."""
        rel, a, b = self.parse_question(question)
        return self.kb.holds(rel, a, b)

    def answer(self, question: str) -> str:
        return "да" if self.ask(question) else "нет"
