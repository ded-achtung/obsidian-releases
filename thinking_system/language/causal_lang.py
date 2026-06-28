"""Понимание причинных утверждений: язык → причинная модель → вмешательства/«что если».

Система разбирает утверждения «X вызывает Y» в причинный граф, строит булеву
причинную модель (следствие истинно, если истинна любая его причина) и отвечает на
вопросы РАССУЖДЕНИЕМ по модели: причинно ли X влияет на Y, что будет, если СДЕЛАТЬ
X (do), и различает «видеть» (корреляция) от «делать» (причинность). Контрфактику —
«что было бы с этим случаем» — считает через ту же модель (SCM из reasoning.causal).

Контролируемая грамматика: «X вызывает Y» / «X приводит к Y» (утверждение, предлог
после маркера пропускается); «X вызывает Y?» (есть ли причинный путь); «если X,
будет Y?» (вмешательство do(X)). За пределами этих шаблонов разбор не гарантируется —
это узкая контролируемая грамматика, а не полный NLP.
"""

from __future__ import annotations

from collections import defaultdict, deque

from thinking_system.reasoning.causal import SCM
from thinking_system.language.understanding import tokenize

_MARK = {"вызывает", "вызывают", "вызвать", "причина", "приводит", "приводят"}
_QSTOP = {"если", "включить", "выключить", "будет", "то", "работает", "ли", "это"}
# предлоги после маркера («приводит К раку») пропускаем, чтобы ребро шло на СЛЕДСТВИЕ, а не на предлог
_PREP = {"к", "ко", "на", "в", "во", "из", "за", "у", "от", "о", "об", "со", "с", "по"}


def _cause_effect(toks: list[str], i: int) -> tuple[str, str]:
    """X и Y вокруг маркера на позиции i, пропуская предлоги (X слева, Y справа)."""
    left = [t for t in toks[:i] if t not in _PREP]
    right = [t for t in toks[i + 1:] if t not in _PREP]
    x = left[-1] if left else toks[i - 1]
    y = right[0] if right else toks[i + 1]
    return x, y


class CausalReader:
    """Читает причинные утверждения и отвечает на вмешательства/«что если»."""

    def __init__(self) -> None:
        self.parents: dict[str, list[str]] = defaultdict(list)
        self.vars: set[str] = set()

    def tell(self, statement: str) -> tuple[str, str]:
        """«X вызывает Y» → причинное ребро X→Y."""
        toks = tokenize(statement)
        i = next(k for k, t in enumerate(toks) if t in _MARK)
        x, y = _cause_effect(toks, i)
        if x not in self.parents[y]:
            self.parents[y].append(x)
        self.vars |= {x, y}
        return (x, y)

    def _order(self) -> list[str]:
        placed: list[str] = []
        remaining = set(self.vars)
        while remaining:
            ready = [v for v in remaining if all(p in placed for p in self.parents.get(v, []))]
            if not ready:
                break                                        # цикл — стоп (причинность ацикл.)
            placed += sorted(ready)
            remaining -= set(ready)
        return placed

    def model(self) -> SCM:
        """Булева причинная модель: корни ~Bern(0.5); следствие = OR его причин."""
        m = SCM()
        for v in self._order():
            par = self.parents.get(v, [])
            if not par:
                m.root(v, {0: 0.5, 1: 0.5})
            else:
                m.eq(v, par, lambda p, par=par: int(any(p[x] for x in par)))
        return m

    def causes(self, x: str, y: str) -> bool:
        """Есть ли ПРИЧИННЫЙ путь X→…→Y (X причинно влияет на Y)."""
        seen = {x}
        q = deque([x])
        child: dict[str, list[str]] = defaultdict(list)
        for c, ps in self.parents.items():
            for p in ps:
                child[p].append(c)
        while q:
            u = q.popleft()
            if u == y and u != x:
                return True
            for v in child.get(u, []):
                if v not in seen:
                    seen.add(v); q.append(v)
        return y in seen and y != x

    def correlated(self, x: str, y: str) -> bool:
        """Связаны ли X и Y наблюдательно (видеть), даже без причинного пути."""
        return abs(self.model().correlation(x, y)) > 1e-9

    def would(self, do_var: str, query: str) -> bool:
        """Вмешательство: если СДЕЛАТЬ do_var, станет ли query истинным?"""
        return self.model().prob(query, 1, do={do_var: 1}) == 1.0

    def counterfactual(self, evidence: dict, do: dict, query: str) -> dict:
        return self.model().counterfactual(evidence, do, query)

    # ── разбор вопросов на естественном языке ────────────────────────────────────
    def ask(self, question: str) -> bool:
        toks = tokenize(question)
        if "если" in toks:                                   # «если X, будет Y?» — вмешательство
            content = [t for t in toks if t not in _QSTOP]
            return self.would(content[0], content[-1])
        i = next((k for k, t in enumerate(toks) if t in _MARK), None)  # «X вызывает Y?»
        if i is None or not 0 < i < len(toks) - 1:
            return False                                     # не распознан причинный вопрос
        x, y = _cause_effect(toks, i)
        return self.causes(x, y)

    def answer(self, question: str) -> str:
        return "да" if self.ask(question) else "нет"
