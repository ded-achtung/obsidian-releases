"""Понимание задачи: язык → смысл → исполнение (а не генерация текста).

Система УЧИТ смысл слова из ПОКАЗА: видит употребление слова в демонстрации
(вход→выход), ИНДУЦИРУЕТ операцию, объясняющую её, и СВЯЗЫВАЕТ слово с этой
операцией. Дальше она ПОНИМАЕТ новые предложения — разбирает их в программу
(композиция значений слов) и ИСПОЛНЯЕТ, выдавая истинный ответ.

Это смысл-как-употребление (Витгенштейн), заземлённый в исполнении (Харнад):
система «понимает» слово, потому что УМЕЕТ делать им обозначенное, обобщает на
новые входы и КОМПОНУЕТ известные слова в новые предложения. Не сэмплирование
правдоподобного — разбор → вывод → исполнение.

Узко и честно: контролируемый словарь (понимает выученные слова; незнакомое —
честно говорит, что не понял), без морфологии, разбор по ключевым словам с
композицией слева-направо.
"""

from __future__ import annotations

import re

from thinking_system.reasoning.induction import Program, default_library

_STOP = {"и", "список", "массив", "числа", "число", "по", "этот", "the", "a", "an", "of", "list"}


def tokenize(text: str) -> list[str]:
    """Слова (кириллица/латиница), числа и скобки игнорируются."""
    return re.findall(r"[а-яёa-z]+", text.lower())


def extract_arg(text: str):
    """Аргумент из предложения: список [..] или одиночное число."""
    m = re.search(r"\[([^\]]*)\]", text)
    if m:
        return [int(x) for x in re.findall(r"-?\d+", m.group(1))]
    m2 = re.search(r"-?\d+", text)
    return int(m2.group()) if m2 else None


class GroundedLexicon:
    """Заземлённый словарь: слово → операция, выученная из показа; разбор и исполнение."""

    def __init__(self, library=None, *, stop: set[str] | None = None) -> None:
        self.lib = library or default_library()
        self.words: dict[str, Program] = {}
        self.stop = stop or set(_STOP)

    def _content(self, sentence: str) -> list[str]:
        return [w for w in tokenize(sentence) if w not in self.stop]

    # ── учить смысл из показа ─────────────────────────────────────────────────────
    def learn(self, word: str, examples: list[tuple]) -> bool:
        """Связать слово с операцией, ИНДУЦИРОВАННОЙ из примеров его употребления."""
        prog = self.lib.induce(examples, max_depth=3)
        if prog is None:
            return False
        self.words[word] = prog
        return True

    def learn_from_demo(self, sentence: str, inp, out) -> str | None:
        """Из демонстрации «предложение + вход→выход» выучить ЕДИНСТВЕННОЕ новое слово."""
        new = [w for w in self._content(sentence) if w not in self.words]
        if len(new) != 1:
            return None                                      # неоднозначно: не одно новое слово
        prog = self.lib.induce([(inp, out)], max_depth=3)
        if prog is None:
            return None
        self.words[new[0]] = prog
        return new[0]

    # ── понимать и решать ────────────────────────────────────────────────────────
    def understand(self, sentence: str) -> dict:
        """Разобрать предложение в программу (значения слов) + аргумент; отметить незнакомое."""
        ops, unknown = [], []
        for w in self._content(sentence):
            (ops if w in self.words else unknown).append(w)
        return {"ops": ops, "arg": extract_arg(sentence), "unknown": unknown}

    def solve(self, sentence: str) -> dict:
        """Понять → ИСПОЛНИТЬ (слова применяются слева-направо) → истинный ответ."""
        u = self.understand(sentence)
        if u["unknown"]:
            return {"understood": False, "unknown": u["unknown"]}
        x = u["arg"]
        for w in u["ops"]:
            x = self.words[w](x)
        return {"understood": True, "answer": x, "reasoning": " ▸ ".join(u["ops"])}
