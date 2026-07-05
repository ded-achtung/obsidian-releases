"""Единый постоянный агент (Mind): сам маршрутизирует опыт, копит знание между запусками.

Закрывает главный разрыв прототипа «думают компоненты — не думает система»:

  • ОДИН субъект: перцепция типа опыта (текст / грид-задача) и выбор способности —
    внутри агента, а не в скрипте-оркестраторе;
  • перенос ЧЕРЕЗ ГРАНИЦУ ДОМЕНОВ: слова учебника заземляются в грид-примитивы
    индукцией из показов, определения растят библиотеку — выигрыш меряется на
    скрытых test-входах реального ARC;
  • ДЕЛИБЕРАЦИЯ: бюджет поиска эскалирует с трудностью (лестница глубина/бюджет),
    вместо фиксированного; честное «не решил» при исчерпании;
  • ПАМЯТЬ МЕЖДУ ЗАПУСКАМИ: библиотека, словарь, решения и нерешённое сохраняются
    в JSON и переживают процесс;
  • ПОВЕСТКА (зачаток автотелии): агент сам возвращается к нерешённым задачам,
    когда его язык вырос, и консолидирует повторяющиеся комбо из своих решений.

Честные границы: словарь закрытый (учебник контролируемый), абстракции — линейные
композиции без переменных, «повестка» — простая очередь. Это интеграция, а не AGI.
"""

from __future__ import annotations

import json
import os
from collections import Counter

from thinking_system.mind.lexicon import (GridLexicon, definition_gaps, parse_grid_alias,
                                          parse_grid_definition, parse_grid_demo)
from thinking_system.reasoning import object_param, parametric
from thinking_system.reasoning.deep_search import bigram_prior, guided_induce
from thinking_system.reasoning.grid_seed import guard, guarded_grid_seed
from thinking_system.reasoning.grids import Grid, to_grid
from thinking_system.reasoning.induction import Primitive, Program
from thinking_system.reasoning.search_prior import best_first_induce
from thinking_system.reasoning.templates import anti_unify, template_search

# лестница размышления: (глубина, бюджет программ); дальше по лестнице = думать дольше;
# ступени глубины 3+ идут УМНЫМ поиском (биграммный приор из опыта + эвристика цели)
LADDER = [(1, 600), (2, 20000), (3, 40000)]


class Mind:
    """Постоянный агент над сетками и текстом с растущей библиотекой и словарём."""

    def __init__(self, state_path: str | None = None) -> None:
        self.prims: list[Primitive] = guarded_grid_seed()
        self._seed_names = {p.name for p in self.prims}
        self.abstractions: list[str] = []                    # канонические имена ("a∘b")
        self.lexicon = GridLexicon(self.prims)
        self.solutions: dict[str, list[str]] = {}            # задача → имена шагов решения
        self.unsolved: dict[str, list] = {}                  # задача → train-пары (для повестки)
        self.questions: list[str] = []                       # слова, которые встретил, но не заземлил
        self.texts_read: list[str] = []
        self._dirty_since_retry = False
        self.state_path = state_path
        if state_path and os.path.exists(state_path):
            self._load(state_path)

    # ── восприятие и маршрутизация: агент сам решает, какую способность применить ──

    @staticmethod
    def perceive(item) -> str:
        """Тип опыта по самому опыту: текст / грид-задача / неизвестное."""
        if isinstance(item, str):
            return "text"
        if isinstance(item, dict) and "train" in item:
            pairs = item["train"]
            if pairs and all(len(p) == 2 for p in pairs):
                return "grid_task"
        return "unknown"

    def experience(self, item, *, effort: int = 2) -> dict:
        """Принять опыт: агент маршрутизирует его сам и возвращает отчёт."""
        kind = self.perceive(item)
        if kind == "text":
            return {"routed": "text", **self.read(item)}
        if kind == "grid_task":
            return {"routed": "grid_task", **self.attempt(item["id"], item["train"], effort=effort)}
        return {"routed": "unknown"}

    # ── способность: чтение (текст → словарь → библиотека) ─────────────────────────

    @staticmethod
    def _scan(text: str) -> tuple[dict, list[str]]:
        """Разбор текста БЕЗ обучения: (показы по словам, прочие строки)."""
        demos: dict[str, list] = {}
        others: list[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            demo = parse_grid_demo(line)
            if demo:
                w, gin, gout = demo
                demos.setdefault(w, []).append((gin, gout))
            else:
                others.append(line)
        return demos, others

    def peek(self, text: str) -> dict:
        """Эпистемическая ценность текста БЕЗ обучения: сколько нового он даст СЕЙЧАС.

        Ценность = новые заземляемые слова (показы) + определения/синонимы,
        собираемые из известного (с учётом слов, которые станут известны из
        показов этого же текста и цепочек определений внутри него).
        """
        demos, others = self._scan(text)
        new_words = [w for w in demos if w not in self.lexicon.words]
        known = set(self.lexicon.words) | set(demos)
        groundable: list[str] = []
        changed = True
        while changed:                                       # определения могут опираться друг на друга
            changed = False
            for line in others:
                parsed = parse_grid_definition(line, known) or parse_grid_alias(line, known)
                if parsed and parsed[0] not in known:
                    known.add(parsed[0])
                    groundable.append(parsed[0])
                    changed = True
        gaps = sorted({w for line in others for w in definition_gaps(line, known)})
        return {"value": len(new_words) + len(groundable),
                "new_words": new_words, "groundable": groundable, "gaps": gaps}

    def read(self, text: str) -> dict:
        """Учебник: показы заземляют слова индукцией, определения/синонимы растят язык."""
        demos, others = self._scan(text)
        learned = [w for w, ex in demos.items() if self.lexicon.learn(w, ex)]
        defined: list[str] = []
        changed = True
        while changed:                                       # цепочки определений внутри текста
            changed = False
            for line in others:
                known = set(self.lexicon.words)
                parsed = parse_grid_definition(line, known)
                if parsed and parsed[0] not in known and self.lexicon.define(*parsed):
                    defined.append(parsed[0])
                    self._add_abstraction(self.lexicon.words[parsed[0]])
                    changed = True
                    continue
                alias = parse_grid_alias(line, known)
                if alias and self.lexicon.define(alias[0], [alias[1]]):
                    defined.append(alias[0])
                    changed = True
        known = set(self.lexicon.words)
        gaps = {w for line in others for w in definition_gaps(line, known)}
        self.questions = sorted((set(self.questions) | gaps) - known)  # выученное — не вопрос
        self.texts_read.append(text[:60])
        if defined or learned:
            self._dirty_since_retry = True
        return {"выучено_слов": learned, "определено": defined,
                "вопросы": sorted(gaps - known)}

    def study_library(self, library: dict[str, str]) -> list[dict]:
        """ЛЮБОПЫТСТВО над библиотекой: читать в порядке эпистемической ценности.

        На каждом шаге агент заново оценивает непрочитанные тексты (чтение
        одного меняет ценность других — куррикулум возникает сам) и честно
        останавливается, когда выучить больше нечего.
        """
        unread = dict(library)
        log: list[dict] = []
        while unread:
            peeks = {t: self.peek(x) for t, x in unread.items()}
            best = max(peeks, key=lambda t: peeks[t]["value"])
            if peeks[best]["value"] == 0:
                log.append({"пропущено": sorted(unread), "причина": "ничего выучить"})
                break
            log.append({"выбрано": best, "ценность": peeks[best]["value"],
                        **self.read(unread.pop(best))})
        return log

    # ── способность: решение грид-задачи с эскалацией размышления ──────────────────

    def attempt(self, task_id: str, train_pairs: list, *, effort: int = 2) -> dict:
        """Искать программу по train-парам: лестница бюджета, затем шаблоны с дыркой.

        Переменные в двух видах: параметрические примитивы (аргумент-цвет связан
        из палитры ЭТОЙ задачи) участвуют во всех ступенях; если лестница
        исчерпана — шаблоны, анти-унифицированные из накопленных решений
        (переменная = целый шаг), делают глубокие структуры достижимыми дёшево.
        """
        pairs = [(to_grid(i) if not isinstance(i, tuple) else i,
                  to_grid(o) if not isinstance(o, tuple) else o) for i, o in train_pairs]
        prims = self.prims + [guard(p) for p in
                              parametric.instantiate(pairs) + object_param.instantiate()
                              + object_param.instantiate_predicates(pairs)]
        weights = self._weights()
        checked_total = 0
        for depth, budget in LADDER[:effort]:
            if depth >= 3:               # глубоко = умно: биграммы опыта + эвристика цели
                prog, n = guided_induce(pairs, prims, bigram_prior(list(self.solutions.values())),
                                        max_depth=depth, budget=budget)
            else:
                prog, n = best_first_induce(pairs, prims, weights,
                                            max_depth=depth, budget=budget)
            checked_total += n
            if prog is not None:
                return self._solved(task_id, prog, checked_total, {"depth": depth})
        templates = anti_unify(list(self.solutions.values()))
        if templates:
            prog, n = template_search(pairs, templates, prims, resolve=self._resolve)
            checked_total += n
            if prog is not None:
                return self._solved(task_id, prog, checked_total, {"via": "template"})
        self.unsolved[task_id] = [[list(map(list, i)), list(map(list, o))] for i, o in pairs]
        return {"solved": False, "checked": checked_total}

    def _solved(self, task_id: str, prog: Program, checked: int, extra: dict) -> dict:
        self.solutions[task_id] = [s.name for s in prog.steps]
        self.unsolved.pop(task_id, None)
        return {"solved": True, "program": prog, "checked": checked, **extra}

    def program_for(self, task_id: str) -> Program | None:
        names = self.solutions.get(task_id)
        if names is None:
            return None
        steps = [self._resolve(n) for n in names]
        return None if any(s is None for s in steps) else Program(steps)

    # ── повестка: агент сам выбирает, о чём думать дальше ─────────────────────────

    def agenda(self) -> list[str]:
        plan = []
        if self.unsolved and self._dirty_since_retry:
            plan.append(f"вернуться к нерешённому ({len(self.unsolved)}) — язык вырос")
        if self._repeated_combos():
            plan.append("консолидировать повторяющиеся комбо из своих решений")
        if self.questions:
            plan.append(f"открытые вопросы ({len(self.questions)}): "
                        + ", ".join(f"что такое «{w}»" for w in self.questions[:3]))
        return plan

    def idle_work(self, *, effort: int = 2) -> dict:
        """Поработать по собственной повестке; вернуть отчёт о сделанном."""
        report: dict = {"agenda": self.agenda(), "resolved": [], "consolidated": []}
        if self.unsolved and self._dirty_since_retry:
            self._dirty_since_retry = False
            for tid, pairs in list(self.unsolved.items()):
                res = self.attempt(tid, pairs, effort=effort)
                if res["solved"]:
                    report["resolved"].append((tid, str(res["program"]), res["checked"]))
        for combo in self._repeated_combos():
            self._add_abstraction(list(combo))
            report["consolidated"].append("∘".join(combo))
        return report

    # ── память между запусками ─────────────────────────────────────────────────────

    def save(self, path: str | None = None) -> str:
        path = path or self.state_path
        assert path, "нужен путь состояния"
        state = {"abstractions": [n.split("∘") for n in self.abstractions],
                 "lexicon": self.lexicon.words,
                 "solutions": self.solutions,
                 "unsolved": self.unsolved,
                 "questions": self.questions,
                 "texts_read": self.texts_read}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=1)
        return path

    def _load(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        for seq in state["abstractions"]:
            self._add_abstraction(seq)
        self.lexicon.words.update(state["lexicon"])
        self.solutions.update(state["solutions"])
        self.unsolved.update(state["unsolved"])
        self.questions = state.get("questions", [])
        self.texts_read = state.get("texts_read", [])

    # ── внутреннее ─────────────────────────────────────────────────────────────────

    def _resolve(self, name: str) -> Primitive | None:
        """Имя → примитив: библиотека агента, параметрическое или объектное семейство."""
        for p in self.prims:
            if p.name == name:
                return p
        p = parametric.by_name(name) or object_param.by_name(name)
        return guard(p) if p is not None else None

    def _add_abstraction(self, step_names: list[str]) -> bool:
        """Композиция имён → новый примитив «a∘b» (если исполнима и нова)."""
        name = "∘".join(step_names)
        if name in {p.name for p in self.prims}:
            return False
        steps = [self._resolve(n) for n in step_names]
        if any(s is None for s in steps):
            return False
        prog = Program(steps)
        self.prims.append(Primitive(name, prog.__call__, cost=1.0))
        self.abstractions.append(name)
        self.lexicon._by_name[name] = self.prims[-1]
        return True

    def _weights(self) -> dict[str, float] | None:
        """Приор поиска из СОБСТВЕННОГО опыта: чаще выручавшие примитивы — раньше."""
        if not self.solutions:
            return None
        c = Counter(n for names in self.solutions.values() for n in names)
        w = {p.name: c.get(p.name, 0) + 1.0 for p in self.prims}
        for n, k in c.items():                               # параметрические из опыта
            w.setdefault(n, k + 1.0)
        return w

    def _repeated_combos(self) -> list[tuple]:
        """Смежные пары шагов, повторившиеся в ≥2 решениях и ещё не названные."""
        counts: Counter = Counter()
        for names in self.solutions.values():
            for i in range(len(names) - 1):
                counts[tuple(names[i:i + 2])] += 1
        existing = {p.name for p in self.prims}
        return [seq for seq, k in counts.items()
                if k >= 2 and "∘".join(seq) not in existing]
