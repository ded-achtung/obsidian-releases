"""Researcher — замыкает исследовательский круг над прочитанным знанием.

Прочитав материал (LessonReader), система не ждёт вопросов, а САМА:

  • ВЫВОДИТ новое — факты, которые следуют из прочитанного, но не были сказаны;
  • СТАВИТ и ПРОВЕРЯЕТ гипотезы — свойства операций ИСПОЛНЕНИЕМ (напр. инволюция),
    причинные связи по модели (включая «связаны, но не причина»);
  • НАХОДИТ ПРОБЕЛЫ — явления без объяснённой причины («что вызывает X?»);
  • ВСТРАИВАЕТ новое знание — дочитав ответ, переоткрывает следствия.

Это разбор→вывод→проверка→поиск пробелов→интеграция, а не генерация. Цикл
замыкается (нашёл пробел → узнал → вывел новое) и открывается снова — система тем
способнее, чем больше прочитала и обдумала.
"""

from __future__ import annotations

from thinking_system.language.reader import LessonReader
from thinking_system.language.causal_lang import _MARK
from thinking_system.language.facts import _UNIV
from thinking_system.language.understanding import tokenize


class Researcher:
    """Автономный исследователь над знанием, прочитанным из текста."""

    def __init__(self, reader: LessonReader) -> None:
        self.r = reader

    # ── вывести новое знание (то, что следует, но не сказано) ─────────────────────
    def derive_new_facts(self) -> list[tuple[str, str, str]]:
        kb = self.r.facts.kb
        return sorted(f for f in kb.closure() if f not in kb.facts)

    # ── проверить гипотезы о свойствах операций ИСПОЛНЕНИЕМ ───────────────────────
    def operation_properties(self, samples: list) -> dict[str, str]:
        out: dict[str, str] = {}
        for w, prog in self.r.lex.words.items():
            involution = True
            applicable = True
            for x in samples:
                xv = list(x) if isinstance(x, list) else x
                try:
                    twice = prog(prog(xv))
                except (TypeError, ValueError, IndexError, KeyError):  # выход не того типа → нельзя применить дважды
                    applicable = False
                    break
                involution = involution and (twice == xv)
            if not applicable:
                out[w] = "нельзя применить дважды"
            else:
                out[w] = "инволюция (применить дважды = вернуть исходное)" if involution else "не инволюция"
        return out

    # ── проанализировать причинную структуру ─────────────────────────────────────
    def causal_findings(self) -> dict:
        cr = self.r.causal
        vs = sorted(cr.vars)
        causes = [(a, b) for a in vs for b in vs if a != b and cr.causes(a, b)]
        corr = [(a, b) for i, a in enumerate(vs) for b in vs[i + 1:]
                if not cr.causes(a, b) and not cr.causes(b, a) and cr.correlated(a, b)]
        roots = [v for v in vs if not cr.parents.get(v)]
        return {"causes": causes, "correlated_not_causal": corr, "roots": roots}

    def open_questions(self) -> list[str]:
        """Пробелы как исследовательские вопросы: явления без объяснённой причины."""
        return [f"что вызывает «{v}»?" for v in self.causal_findings()["roots"]]

    # ── встроить новое знание (замкнуть и переоткрыть цикл) ───────────────────────
    def learn_more(self, statement: str) -> str:
        toks = tokenize(statement)
        if any(m in toks for m in _MARK):
            self.r.causal.tell(statement)
            return "причина"
        self.r.facts.tell(statement)
        return "факт"
