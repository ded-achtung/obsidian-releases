"""Заземление слов в ГРИД-примитивы: показ → индукция → слово; определения → композиции.

Мост «язык → сетки»: слово из учебника связывается с примитивом не переводом,
а ИНДУКЦИЕЙ из показа («отражение [[1,2],[3,4]] → [[2,1],[4,3]]» — система сама
находит, какой из её примитивов объясняет пример). Определение («поворот это
сначала отражение потом переворот») компонует уже заземлённые слова. Так знание,
пришедшее ТЕКСТОМ, становится операциями над сетками — и меряется на реальном ARC.

Частично ОТКРЫТЫЙ словарь: формы слова сводятся к основе стеммером
(language/morphology) — «отражение / отражением / отражения» узнаются как одно
слово, определения можно писать естественными падежами. Показ распознаётся и
внутри свободной прозы: строка с РОВНО двумя сетками читается как «вход → выход»,
слово-операция — ближайшее содержательное слово перед первой сеткой. Честные
границы: стеммер снимает только словоизменение (не словообразование), проза —
на уровне строк-предложений, не абзацев со ссылками.
"""

from __future__ import annotations

import ast
import re

from thinking_system.language.morphology import stem
from thinking_system.reasoning.grids import Grid, to_grid
from thinking_system.reasoning.induction import Primitive, Program, induce

_GRID_RE = re.compile(r"\[\s*\[[^\[\]]*\](?:\s*,\s*\[[^\[\]]*\])*\s*\]")
_SEP_RE = re.compile(r"\s*(?:=|даёт|дает|равно|→|->)\s*")
_WORD_RE = re.compile(r"[а-яёa-z]+")
_SEQ = {"сначала", "потом", "затем", "после"}
_STOP = {"это", "и", "в", "на", "сетка", "сетку", "сетки", "пример", "например",
         "ещё", "еще", "превращает", "становится", "всегда"} | _SEQ


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _content(tokens: list[str]) -> list[str]:
    return [w for w in tokens if w not in _STOP]


def stems_match(a: str, b: str) -> bool:
    """Основы совпадают или соотносятся как префиксы (≥4 символов).

    Стеммер снимает разные окончания у разных форм («отражение»→«отражен»,
    «отражения»→«отражени») — префиксное сопоставление сшивает такие пары,
    не склеивая короткие случайные совпадения."""
    if a == b:
        return True
    if min(len(a), len(b)) < 4:
        return False
    return a.startswith(b) or b.startswith(a)


def known_has(word: str, known_stems: set[str]) -> bool:
    s = stem(word)
    return any(stems_match(s, k) for k in known_stems)


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
    """Показ «<слово> [[вход]] → [[выход]]» → (слово, вход, выход) или None.

    Работает и в прозе: строка с РОВНО двумя сетками читается как вход→выход,
    слово-операция — последнее содержательное слово ПЕРЕД первой сеткой.
    """
    parts = _SEP_RE.split(line, maxsplit=1)
    if len(parts) == 2:
        left, right = parts
        gin, gout = extract_grids(left), extract_grids(right)
        if len(gin) == 1 and len(gout) == 1:
            words = _content(tokenize(left))
            if words:
                return words[-1], gin[0], gout[0]
    grids = extract_grids(line)                              # fallback: проза без «→»
    if len(grids) == 2:
        first = _GRID_RE.search(line)
        words = _content(tokenize(line[: first.start()]))
        if words:
            return words[-1], grids[0], grids[1]
    return None


def parse_grid_definition(line: str, known_stems: set[str]):
    """«<новое> это сначала <A> потом <B> …» → (новое, [известные слова]) или None.

    Известность слова проверяется по ОСНОВЕ — определения можно писать падежами
    («…сначала отражения потом переворота»)."""
    toks = tokenize(line)
    if "это" not in toks:
        return None
    i = toks.index("это")
    left, right = toks[:i], toks[i + 1:]
    ops = [w for w in right if known_has(w, known_stems)]
    new = [w for w in _content(left) if not known_has(w, known_stems)]
    if len(new) == 1 and len(ops) >= 2 and any(w in _SEQ for w in right):
        return new[0], ops
    return None


def parse_grid_alias(line: str, known_stems: set[str]):
    """«<новое> значит <известное>» → (новое, известное) или None (синоним, 1:1)."""
    toks = tokenize(line)
    if "значит" not in toks:
        return None
    i = toks.index("значит")
    left = [w for w in _content(toks[:i]) if not known_has(w, known_stems)]
    right = [w for w in toks[i + 1:] if known_has(w, known_stems)]
    if len(left) == 1 and len(right) == 1:
        return left[0], right[0]
    return None


def definition_gaps(line: str, known_stems: set[str]) -> list[str]:
    """Незаземлённые слова-операции в строке-определении — открытые ВОПРОСЫ агента."""
    toks = tokenize(line)
    if "это" not in toks or not any(w in _SEQ for w in toks):
        return []                                            # не похоже на определение
    right = toks[toks.index("это") + 1:]
    return [w for w in _content(right) if not known_has(w, known_stems)]


class GridLexicon:
    """Словарь слово → имена примитивов; формы слова узнаются по общей основе."""

    def __init__(self, primitives: list[Primitive]) -> None:
        self._by_name = {p.name: p for p in primitives}
        self.words: dict[str, list[str]] = {}                # каноническое слово → шаги
        self._stems: dict[str, str] = {}                     # основа → каноническое слово

    def known_stems(self) -> set[str]:
        return set(self._stems)

    def resolve(self, word: str) -> str | None:
        """Любая форма слова → каноническое слово (точно или по префиксу основ)."""
        s = stem(word)
        if s in self._stems:
            return self._stems[s]
        hits = {c for k, c in self._stems.items() if stems_match(s, k)}
        return hits.pop() if len(hits) == 1 else None        # неоднозначно — честный отказ

    def learn(self, word: str, examples: list[tuple[Grid, Grid]]) -> bool:
        """Заземлить слово: найти примитив (глубина 1), объясняющий ВСЕ показы."""
        prog = induce(examples, list(self._by_name.values()), max_depth=1)
        if prog is None or not prog.steps:
            return False
        self.words[word] = [s.name for s in prog.steps]
        self._stems[stem(word)] = word
        return True

    def define(self, word: str, op_words: list[str]) -> bool:
        """Новое слово = композиция известных (в любых формах); шаги — базовые имена."""
        canon = [self.resolve(w) for w in op_words]
        if any(c is None for c in canon):
            return False
        self.words[word] = [n for c in canon for n in self.words[c]]
        self._stems[stem(word)] = word
        return True

    def program(self, word: str) -> Program | None:
        canon = self.resolve(word)
        if canon is None:
            return None
        return Program([self._by_name[n] for n in self.words[canon]])
