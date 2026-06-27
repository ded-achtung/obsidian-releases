"""Условные правила: применимость зависит от ДАННЫХ, а не от слов задачи.

Шаг дальше учёбы (study.py): правило выбирается по СВОЙСТВУ самих данных. Система
учит операции из примеров, а из теории — правила вида «если <условие на данных>,
<операция>». При решении она проверяет условия на входе и применяет подходящее
правило. Одна задача «обработай» решается по-разному в зависимости от данных.

Условия — небольшой набор приоров-предикатов над списком (есть отрицательные,
длинный, короткий, пустой); слова теории сопоставляются им. Операции (включая
«очисти» = оставить положительные) учатся из примеров.
"""

from __future__ import annotations

from thinking_system.reasoning.induction import Primitive, Library, default_primitives, _lst
from thinking_system.language.understanding import GroundedLexicon, tokenize, extract_arg
from thinking_system.language.reader import parse_demonstration

# предикаты-приоры над данными + слова, по которым их распознать в теории
_PREDICATES = {
    "отрицательные": lambda xs: any(v < 0 for v in xs),
    "положительные": lambda xs: len(xs) > 0 and all(v > 0 for v in xs),
    "пустой": lambda xs: len(xs) == 0,
    "длинный": lambda xs: len(xs) > 3,
    "короткий": lambda xs: 0 < len(xs) <= 3,
    "чётные": lambda xs: any(v % 2 == 0 for v in xs),
}


def _filter_library() -> Library:
    """Язык операций + фильтры (нужны для условных действий, напр. «очисти»)."""
    extra = [
        Primitive("positives", lambda s: [v for v in _lst(s) if v > 0]),
        Primitive("negatives", lambda s: [v for v in _lst(s) if v < 0]),
    ]
    return Library(default_primitives() + extra)


class ConditionalReader:
    """Учит операции и условные правила (условие-на-данных → операция); решает по данным."""

    def __init__(self) -> None:
        self.lex = GroundedLexicon(_filter_library())
        self.rules: list[tuple[str, str]] = []           # (слово-условие, слово-операция)
        self._demos: dict[str, list] = {}

    def study(self, text: str) -> dict:
        from collections import defaultdict

        self._demos = defaultdict(list)
        theory: list[str] = []
        exercises: list[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            demo = parse_demonstration(line)
            if demo:
                w, inp, out = demo
                self._demos[w].append((inp, out))
            elif line.lower().startswith("задача"):
                exercises.append(line.split(":", 1)[1].strip())
            elif "если" in tokenize(line):
                theory.append(line)
        for w, ex in self._demos.items():
            self.lex.learn(w, ex)
        for line in theory:
            self.learn_rule(line)
        self.exercises = exercises
        return {"операции": len(self.lex.words), "правила": len(self.rules), "задачи": len(exercises)}

    def learn_rule(self, line: str) -> None:
        """«если <условие>, <операция>» → правило (условие-слово, операция-слово)."""
        toks = tokenize(line)
        cond = next((t for t in toks if t in _PREDICATES), None)
        op = next((t for t in toks if t in self.lex.words), None)
        if cond and op:
            self.rules.append((cond, op))

    def solve(self, task: str) -> dict:
        """Выбрать правило по СВОЙСТВУ данных и применить (первое подходящее условие)."""
        data = extract_arg(task)
        if not isinstance(data, list):
            return {"solved": False, "reason": "нет списка-данных"}
        for cond, op in self.rules:
            if _PREDICATES[cond](data):
                return {"solved": True, "answer": self.lex.words[op](data), "rule": f"если {cond} → {op}"}
        return {"solved": False, "reason": "ни одно условие не подошло"}

    def solve_exercises(self) -> list[dict]:
        return [{"задача": t, **self.solve(t)} for t in self.exercises]
