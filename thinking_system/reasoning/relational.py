"""Реляционно-аналоговое рассуждение: крошечный логический движок (мини-Datalog).

Знания — это ОТНОШЕНИЯ: факты-тройки (отношение, a, b). Поверх — правила Хорна с
переменными; прямой вывод (forward chaining) доводит факты до замыкания. Из
НЕСКОЛЬКИХ фактов выводится много новых — без корпуса, чистым выводом.

На этом ядре выражаются:
  • транзитивность  R(x,y)∧R(y,z) ⇒ R(x,z)   (порядок, предки, таксономия);
  • симметрия       R(x,y) ⇒ R(y,x);
  • наследование    is_a(x,y)∧P(y,v) ⇒ P(x,v) (свойства класса переходят на вид);
  • аналогия A:B :: C:? — найти отношение, связывающее A→B, и применить к C.

Переменная — терм, начинающийся с «?»; всё остальное — константа. Отношение тоже
терм (позиция 0), поэтому наследование «по любому свойству» — одно правило.
"""

from __future__ import annotations

from typing import Iterator

Fact = tuple[str, str, str]
Pattern = tuple[str, str, str]


def _is_var(t: str) -> bool:
    return t.startswith("?")


def _unify(pat: Pattern, fact: Fact, binding: dict[str, str]) -> dict[str, str] | None:
    b = dict(binding)
    for p, f in zip(pat, fact):
        if _is_var(p):
            if p in b and b[p] != f:
                return None
            b[p] = f
        elif p != f:
            return None
    return b


def _match(body: list[Pattern], facts: set[Fact], binding: dict[str, str]) -> Iterator[dict[str, str]]:
    if not body:
        yield binding
        return
    head, rest = body[0], body[1:]
    for f in facts:
        b2 = _unify(head, f, binding)
        if b2 is not None:
            yield from _match(rest, facts, b2)


def _subst(pat: Pattern, binding: dict[str, str]) -> Fact:
    return tuple(binding.get(t, t) if _is_var(t) else t for t in pat)  # type: ignore[return-value]


def forward_chain(facts: set[Fact], rules: list[tuple[Pattern, list[Pattern]]], *, max_iters: int = 100) -> set[Fact]:
    """Наименьшее замыкание: повторять применение правил, пока появляются новые факты."""
    facts = set(facts)
    for _ in range(max_iters):
        new: set[Fact] = set()
        for head, body in rules:
            for binding in _match(body, facts, {}):
                fact = _subst(head, binding)
                if fact not in facts:
                    new.add(fact)
        if not new:
            break
        facts |= new
    return facts


class KnowledgeBase:
    """Факты-отношения + правила; вывод до замыкания, запросы и аналогии."""

    def __init__(self) -> None:
        self.facts: set[Fact] = set()
        self.rules: list[tuple[Pattern, list[Pattern]]] = []
        self._closure: set[Fact] | None = None

    def add(self, rel: str, a: str, b: str) -> "KnowledgeBase":
        self.facts.add((rel, a, b))
        self._closure = None
        return self

    def rule(self, head: Pattern, body: list[Pattern]) -> "KnowledgeBase":
        self.rules.append((head, body))
        self._closure = None
        return self

    def transitive(self, rel: str) -> "KnowledgeBase":
        return self.rule((rel, "?x", "?z"), [(rel, "?x", "?y"), (rel, "?y", "?z")])

    def symmetric(self, rel: str) -> "KnowledgeBase":
        return self.rule((rel, "?y", "?x"), [(rel, "?x", "?y")])

    def inherits(self, via: str = "is_a") -> "KnowledgeBase":
        """Свойства класса переходят на вид: via(x,y)∧P(y,v) ⇒ P(x,v)."""
        return self.rule(("?p", "?x", "?v"), [(via, "?x", "?y"), ("?p", "?y", "?v")])

    def closure(self) -> set[Fact]:
        if self._closure is None:
            self._closure = forward_chain(self.facts, self.rules)
        return self._closure

    def holds(self, rel: str, a: str, b: str) -> bool:
        return (rel, a, b) in self.closure()

    def query(self, rel: str, a: str | None = None, b: str | None = None) -> list[Fact]:
        """Все выведенные факты отношения rel, согласованные с заданными a/b."""
        return sorted(f for f in self.closure() if f[0] == rel and (a is None or f[1] == a) and (b is None or f[2] == b))

    def relations(self, a: str, b: str) -> list[str]:
        """Какими отношениями связана пара a→b (по замыканию)."""
        return sorted({f[0] for f in self.closure() if f[1] == a and f[2] == b})

    def analogy(self, a: str, b: str, c: str) -> list[str]:
        """A:B :: C:? — отношением(ями), связывающим A→B, найти D для C→D."""
        cl = self.closure()
        rels = self.relations(a, b)
        out = {f[2] for r in rels for f in cl if f[0] == r and f[1] == c}
        return sorted(out)
