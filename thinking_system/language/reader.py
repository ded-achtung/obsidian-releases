"""Чтение СМЫСЛА из текста: смыкает мост «язык→смысл» с книгой/уроком.

Система сканирует текст и сама извлекает разнотипный материал, направляя каждый
кусок в нужный модуль понимания:

  • проработанный пример «слово вход = выход» → учит значение слова (GroundedLexicon);
  • факт «A это B» / «все B P»               → база знаний (FactReader);
  • причинное «X вызывает Y»                  → причинная модель (CausalReader).

Повествовательные строки, которые не распознаны, честно пропускаются. После чтения
система ИСПОЛЬЗУЕТ прочитанное: решает задачи, отвечает на вопросы дедукцией,
рассуждает о причинах. Это «учиться по книге со смыслом» — узко и честно
(контролируемый текст, разбор по образцам, а не полный NLP).
"""

from __future__ import annotations

import re
from collections import defaultdict

from thinking_system.language.understanding import GroundedLexicon, tokenize, extract_arg
from thinking_system.language.facts import FactReader, _UNIV
from thinking_system.language.causal_lang import CausalReader, _MARK

_SEP = re.compile(r"\s*(?:=|даёт|дает|равно|равна|->|→)\s*")
_DEMO_STOP = {"пример", "ещё", "еще", "список", "массив", "и", "к", "на"}


def parse_demonstration(line: str):
    """Строка «слово вход = выход» → (слово, вход, выход) или None."""
    parts = _SEP.split(line, maxsplit=1)
    if len(parts) != 2:
        return None
    left, right = parts
    inp, out = extract_arg(left), extract_arg(right)
    if inp is None or out is None:
        return None
    words = [w for w in tokenize(left) if w not in _DEMO_STOP]
    if not words:
        return None
    return words[-1], inp, out                               # слово-операция — ближайшее к аргументу


class LessonReader:
    """Читает урок-текст, извлекает смысл по типам и даёт пользоваться прочитанным."""

    def __init__(self) -> None:
        self.lex = GroundedLexicon()
        self.facts = FactReader()
        self.causal = CausalReader()
        self._demos: dict[str, list] = defaultdict(list)
        self.stats = {"примеры": 0, "слова": 0, "факты": 0, "причины": 0, "пропущено": 0}

    def read(self, text: str) -> dict:
        """Прочитать текст: распознать тип каждой строки и направить в нужный модуль."""
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            toks = tokenize(line)
            demo = parse_demonstration(line)
            if demo:
                word, inp, out = demo
                self._demos[word].append((inp, out))
                self.stats["примеры"] += 1
            elif any(m in toks for m in _MARK):
                self.causal.tell(line)
                self.stats["причины"] += 1
            elif "это" in toks or any(u in toks for u in _UNIV):
                self.facts.tell(line)
                self.stats["факты"] += 1
            else:
                self.stats["пропущено"] += 1
        for word, examples in self._demos.items():            # выучить значение каждого слова из его примеров
            if self.lex.learn(word, examples):
                self.stats["слова"] += 1
        return self.stats

    def read_file(self, path: str) -> dict:
        from thinking_system.text.ingest import load_book

        return self.read(load_book(path))

    # ── пользоваться прочитанным ─────────────────────────────────────────────────
    def solve(self, task: str) -> dict:
        return self.lex.solve(task)

    def ask_fact(self, question: str) -> str:
        return self.facts.answer(question)

    def ask_causal(self, question: str) -> str:
        return self.causal.answer(question)
