"""Учёба по учебнику: выучить операции, теорию применимости и решать задачи.

Учебник из трёх частей, и система берёт из каждой своё:
  • ПРИМЕРЫ  — что делает операция     → значение слова (индукция, GroundedLexicon);
  • ТЕОРИЯ   — КОГДА её применять       → применимость: слова-приметы → операция;
  • ЗАДАЧИ   — решить, ВЫБРАВ правило   → по приметам задачи подобрать операцию.

Ключевое: в задаче операция НЕ названа («найди ОБЩЕЕ» вместо «сумма») — система
сама РАСПОЗНАЁТ, какое правило применить, по теории применимости, выученной из
учебника. Это «знать, когда использовать знания», а не только что они делают.

ГРАНИЦЫ: «применимость» — это таблица «слово-примета → операция», собранная из слов
СТРОК ТЕОРИИ. Распознавание = поиск этих ТОЧНЫХ примет в задаче; синоним, которого нет
в теории (и форма в другом падеже без морфологии), не сработает. Это запомненные
приметы, а не понимание условия.
"""

from __future__ import annotations

import re
from collections import defaultdict

from thinking_system.language.understanding import GroundedLexicon, tokenize, extract_arg
from thinking_system.language.reader import parse_demonstration

# служебные слова теории — не приметы (общие слова «когда применять»)
_APPLIC_STOP = {"нужна", "нужен", "нужно", "когда", "просят", "применяй", "применять",
                "или", "это", "если", "чтобы", "надо", "значит"}


class Textbook:
    """Учится по учебнику: операции из примеров, применимость из теории, решает задачи."""

    def __init__(self, *, normalize=None) -> None:
        self.norm = normalize or (lambda w: w)         # морфология: форма → основа
        self.lex = GroundedLexicon(normalize=self.norm)
        self.applicability: dict[str, str] = {}        # основа-примета → операция
        self.exercises: list[str] = []
        self._demos: dict[str, list] = defaultdict(list)
        self._applic_stop = {self.norm(w) for w in _APPLIC_STOP}

    def study(self, text: str) -> dict:
        """Прочитать учебник: примеры → операции, теория → применимость, задачи → список."""
        theory: list[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            demo = parse_demonstration(line)
            if demo:
                w, inp, out = demo
                self._demos[w].append((inp, out))
            elif line.lower().startswith(("задача", "упражнение")):
                self.exercises.append(re.sub(r"^[^:]*:\s*", "", line))
            else:
                theory.append(line)
        for w, ex in self._demos.items():               # сперва выучить, ЧТО делают операции
            self.lex.learn(w, ex)
        for line in theory:                             # затем — КОГДА их применять
            self.learn_applicability(line)
        return {"операции": len(self.lex.words), "приметы": len(self.applicability),
                "задачи": len(self.exercises)}

    def learn_applicability(self, line: str) -> None:
        """Из теории связать слова-приметы (по основе) с операцией, упомянутой в строке."""
        toks = [self.norm(t) for t in tokenize(line)]
        ops = [w for w in toks if w in self.lex.words]
        if not ops:
            return
        op = ops[0]
        for w in toks:
            if w not in self.lex.words and w not in self._applic_stop:
                self.applicability[w] = op

    def _ops(self, task: str) -> list[str]:
        """Какие операции применить к задаче: по названию ИЛИ по примете (по основе, в порядке слов)."""
        seq = []
        for t in tokenize(task):
            w = self.norm(t)
            if w in self.lex.words:
                seq.append(w)
            elif w in self.applicability:
                seq.append(self.applicability[w])
        return [o for i, o in enumerate(seq) if i == 0 or o != seq[i - 1]]

    def solve(self, task: str) -> dict:
        """Выбрать правило(а) по смыслу задачи и ИСПОЛНИТЬ."""
        ops, arg = self._ops(task), extract_arg(task)
        if not ops or arg is None:
            return {"solved": False, "reason": "не понял, какое правило применить"}
        x = arg
        for o in ops:
            x = self.lex.words[o](x)
        return {"solved": True, "answer": x, "used": ops}

    def solve_exercises(self) -> list[dict]:
        return [{"задача": t, **self.solve(t)} for t in self.exercises]
