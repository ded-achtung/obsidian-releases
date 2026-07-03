"""Заземление слов в ГРИД-примитивы: показ → индукция → слово; определение → композиция.

Мост «язык → сетки»: слово из учебника связывается с примитивом не переводом,
а ИНДУКЦИЕЙ из показа («отражение [[1,2],[3,4]] → [[2,1],[4,3]]» — система сама
находит, какой из её примитивов объясняет пример). Определение («поворот это
сначала отражение потом переворот») компонует уже заземлённые слова. Так знание,
пришедшее ТЕКСТОМ, становится операциями над сетками — и меряется на реальном ARC.
"""

from __future__ import annotations

import ast
import re

from thinking_system.reasoning.grids import Grid, to_grid
from thinking_system.reasoning.induction import Primitive, Program, induce

_GRID_RE = re.compile(r"\[\s*\[[^\[\]]*\](?:\s*,\s*\[[^\[\]]*\])*\s*\]")
_SEP_RE = re.compile(r"\s*(?:=|даёт|дает|равно|→|->)\s*")
_WORD_RE = re.compile(r"[а-яёa-z]+")
_SEQ = {"сначала", "потом", "затем", "после"}
_STOP = {"это", "и", "сетка", "сетку", "сетки", "пример", "ещё", "еще"} | _SEQ


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def extract_grids(text: str) -> list[Grid]:
    """Все сеточные литералы [[...],[...]] в строке → Grid (кортежи кортежей int)."""
    grids = []
    for m in _GRID_RE.findall(text):
        try:
            rows = ast.literal_eval(m)
        except (ValueError, SyntaxError):
            continue
        if (isinstance(rows, list) and rows and all(isinstance(r, list) for r in rows)
                and all(isinstance(v, int) for r in rows for v in r)):
            grids.append(to_grid(rows))
    return grids


def parse_grid_demo(line: str):
    """«<слово> [[вход]] → [[выход]]» → (слово, вход, выход) или None."""
    parts = _SEP_RE.split(line, maxsplit=1)
    if len(parts) != 2:
        return None
    left, right = parts
    gin, gout = extract_grids(left), extract_grids(right)
    if len(gin) != 1 or len(gout) != 1:
        return None
    words = [w for w in tokenize(left) if w not in _STOP]
    if not words:
        return None
    return words[-1], gin[0], gout[0]                        # слово-операция — ближайшее к сетке


def parse_grid_definition(line: str, known: set[str]):
    """«<новое> это сначала <A> потом <B> …» → (новое, [известные слова]) или None."""
    toks = tokenize(line)
    if "это" not in toks:
        return None
    i = toks.index("это")
    left, right = toks[:i], toks[i + 1:]
    ops = [w for w in right if w in known]
    new = [w for w in left if w not in known and w not in _STOP]
    if len(new) == 1 and len(ops) >= 2 and any(w in _SEQ for w in right):
        return new[0], ops
    return None


class GridLexicon:
    """Словарь слово → последовательность ИМЁН примитивов (заземлено индукцией)."""

    def __init__(self, primitives: list[Primitive]) -> None:
        self._by_name = {p.name: p for p in primitives}
        self.words: dict[str, list[str]] = {}

    def learn(self, word: str, examples: list[tuple[Grid, Grid]]) -> bool:
        """Заземлить слово: найти примитив (глубина 1), объясняющий ВСЕ показы."""
        prog = induce(examples, list(self._by_name.values()), max_depth=1)
        if prog is None or not prog.steps:
            return False
        self.words[word] = [s.name for s in prog.steps]
        return True

    def define(self, word: str, op_words: list[str]) -> bool:
        """Новое слово = композиция известных (имена шагов разворачиваются до базовых)."""
        if any(w not in self.words for w in op_words):
            return False
        self.words[word] = [n for w in op_words for n in self.words[w]]
        return True

    def program(self, word: str) -> Program | None:
        if word not in self.words:
            return None
        return Program([self._by_name[n] for n in self.words[word]])
