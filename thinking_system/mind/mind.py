"""Единый постоянный агент (Mind): сам маршрутизирует опыт, копит знание между запусками.

Закрывает главный разрыв прототипа «думают компоненты — не думает система»:

  • ОДИН субъект: перцепция типа опыта (текст / грид-задача / мир-лабиринт) и выбор
    способности — внутри агента, а не в скрипте-оркестраторе;
  • перенос ЧЕРЕЗ ГРАНИЦУ ДОМЕНОВ: слова учебника заземляются в грид-примитивы
    индукцией из показов, определения растят библиотеку — выигрыш меряется на
    скрытых test-входах реального ARC;
  • ДЕЛИБЕРАЦИЯ: бюджет поиска эскалирует с трудностью (лестница глубина/бюджет),
    вместо фиксированного; честное «не решил» при исчерпании;
  • ПАМЯТЬ МЕЖДУ ЗАПУСКАМИ: библиотека, словарь, решения и нерешённое сохраняются
    в JSON и переживают процесс;
  • ПОВЕСТКА (зачаток автотелии): агент сам возвращается к нерешённым задачам,
    когда его язык вырос, и консолидирует повторяющиеся комбо из своих решений.

Честные границы: словарь частично открытый — формы слова узнаются по основе,
показ читается из свободной прозы, но сами тексты — контролируемые учебники;
абстракции — линейные композиции без переменных, «повестка» — простая очередь.
Это интеграция, а не AGI.
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

# лестница ИССЛЕДОВАНИЯ мира: (эпизодов, шагов/эпизод); беглый взгляд → долгое
# исследование (конфигурация верхней ступени — как в run_world, 10/10 на свежих)
WORLD_LADDER = [(3, 60), (40, 1500)]


class Mind:
    """Постоянный агент над сетками и текстом с растущей библиотекой и словарём."""

    def __init__(self, state_path: str | None = None) -> None:
        self.prims: list[Primitive] = guarded_grid_seed()
        self._seed_names = {p.name for p in self.prims}
        self.abstractions: list[str] = []                    # канонические имена ("a∘b")
        self.lexicon = GridLexicon(self.prims)
        self.solutions: dict[str, list[str]] = {}            # задача → имена шагов решения
        self.unsolved: dict[str, list] = {}                  # задача → train-пары (для повестки)
        self.world_maps: dict[str, list] = {}                # мир → выученные переходы [s,a,s']
        self.unsolved_worlds: dict[str, dict] = {}           # мир → спецификация (для повестки)
        self.questions: list[str] = []                       # слова, которые встретил, но не заземлил
        self.texts_read: list[str] = []
        self._dirty_since_retry = False
        self.state_path = state_path
        if state_path and os.path.exists(state_path):
            self._load(state_path)

    # ── восприятие и маршрутизация: агент сам решает, какую способность применить ──

    @staticmethod
    def perceive(item) -> str:
        """Тип опыта по самому опыту: текст / грид-задача / мир / неизвестное."""
        if isinstance(item, str):
            return "text"
        if isinstance(item, dict) and "world" in item:
            return "world"
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
        if kind == "world":
            return {"routed": "world", **self.explore(item["id"], item["world"], effort=effort)}
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
        from thinking_system.language.morphology import stem

        demos, others = self._scan(text)
        new_words = [w for w in demos if self.lexicon.resolve(w) is None]
        known = self.lexicon.known_stems() | {stem(w) for w in demos}
        from thinking_system.mind.lexicon import stems_match  # noqa: PLC0415
        groundable: list[str] = []
        changed = True
        while changed:                                       # определения могут опираться друг на друга
            changed = False
            for line in others:
                parsed = parse_grid_definition(line, known) or parse_grid_alias(line, known)
                if parsed and not any(stems_match(stem(parsed[0]), k) for k in known):
                    known.add(stem(parsed[0]))
                    groundable.append(parsed[0])
                    changed = True
        gaps = sorted({w for line in others for w in definition_gaps(line, known)})
        return {"value": len(new_words) + len(groundable),
                "new_words": new_words, "groundable": groundable, "gaps": gaps}

    def read(self, text: str) -> dict:
        """Учебник: показы заземляют слова индукцией, определения/синонимы растят язык."""
        from thinking_system.language.morphology import stem

        demos, others = self._scan(text)
        learned = [w for w, ex in demos.items() if self.lexicon.learn(w, ex)]
        defined: list[str] = []
        changed = True
        while changed:                                       # цепочки определений внутри текста
            changed = False
            for line in others:
                known = self.lexicon.known_stems()
                parsed = parse_grid_definition(line, known)
                if parsed and self.lexicon.define(*parsed):
                    defined.append(parsed[0])
                    self._add_abstraction(self.lexicon.words[parsed[0]])
                    changed = True
                    continue
                alias = parse_grid_alias(line, known)
                if alias and self.lexicon.define(alias[0], [alias[1]]):
                    defined.append(alias[0])
                    changed = True
        from thinking_system.mind.lexicon import known_has

        known = self.lexicon.known_stems()
        gaps = {w for line in others for w in definition_gaps(line, known)}
        open_qs = {q for q in set(self.questions) | gaps
                   if not known_has(q, known)}               # выученное — не вопрос
        self.questions = sorted(open_qs)
        self.texts_read.append(text[:60])
        if defined or learned:
            self._dirty_since_retry = True
        return {"выучено_слов": learned, "определено": defined,
                "вопросы": sorted(g for g in gaps if not known_has(g, known))}

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

    # ── способность: мир-лабиринт (активный вывод с лестницей исследования) ────────

    @staticmethod
    def _make_world(spec: dict):
        from thinking_system.world.gridworld import GridWorld

        return GridWorld(int(spec["size"]), {tuple(w) for w in spec["walls"]},
                         start=tuple(spec["start"]), goal=tuple(spec["goal"]))

    @staticmethod
    def _walk(env, agent, *, max_len: int = 200) -> int | None:
        """Пройти к цели РЕАЛЬНО в среде по выученной карте; шагов или None.

        Решённость мира проверяется исполнением, а не самоотчётом модели:
        на каждом шаге берётся действие, сокращающее выученное расстояние
        до цели, и исполняется в среде."""
        s = env.reset()
        for t in range(1, max_len + 1):
            dist = agent.dist_to_goal()
            best_a, best_d = None, float("inf")
            for a in range(env.n_actions):
                sp = agent.model.get((s, a))
                if sp is not None and sp in dist and dist[sp] < best_d:
                    best_d, best_a = dist[sp], a
            if best_a is None:
                return None                                  # карта не ведёт к цели
            s, done = env.step(best_a)
            if done:
                return t
        return None

    def explore(self, world_id: str, spec: dict, *, effort: int = 2) -> dict:
        """Мир: исследовать по лестнице, выучить карту, дойти до цели.

        Способность — существующий ActingAgent (активный вывод: прагматика/
        эпистемика); модель мира — таблица переходов (s,a)→s'. Карта живёт в
        памяти агента и переживает запуск (известный мир решается сразу, без
        исследования). Честные границы: карта ПРО-лабиринтная, переноса между
        мирами нет (это предмет run_transfer); мера здесь — маршрутизация
        третьего типа опыта, эскалация исследования и память."""
        from thinking_system.agent.acting import ActingAgent

        env = self._make_world(spec)
        agent = ActingAgent(env.n_actions, env.goal_state, seed=0)
        for s, a, sp in self.world_maps.get(world_id, []):   # карта прежних сессий
            agent.model[(s, a)] = sp
        explored = 0
        for episodes, max_steps in WORLD_LADDER[:max(effort, 1)]:
            if self._walk(env, agent) is not None:
                break                                        # карта уже ведёт к цели
            for ep in range(episodes):
                eps = max(0.05, 0.5 * (0.88 ** ep))          # любопытство угасает
                s = env.reset()
                for _ in range(max_steps):
                    a = agent.act(s, epsilon=eps)
                    sp, done = env.step(a)
                    agent.learn(s, a, sp)
                    explored += 1
                    s = sp
                    if done:
                        break
        steps = self._walk(env, agent)
        if steps is not None:
            self.world_maps[world_id] = [[s, a, sp] for (s, a), sp in sorted(agent.model.items())]
            self.unsolved_worlds.pop(world_id, None)
            return {"solved": True, "steps": steps, "optimal": env.optimal_steps(),
                    "explored": explored}
        self.unsolved_worlds[world_id] = spec
        return {"solved": False, "explored": explored}

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
        if self.unsolved_worlds:
            plan.append(f"вернуться к неисследованным мирам ({len(self.unsolved_worlds)}) "
                        f"— исследовать дольше")
        if self._repeated_combos():
            plan.append("консолидировать повторяющиеся комбо из своих решений")
        if self.questions:
            plan.append(f"открытые вопросы ({len(self.questions)}): "
                        + ", ".join(f"что такое «{w}»" for w in self.questions[:3]))
        return plan

    def idle_work(self, *, effort: int = 2) -> dict:
        """Поработать по собственной повестке; вернуть отчёт о сделанном."""
        report: dict = {"agenda": self.agenda(), "resolved": [], "consolidated": [],
                        "worlds_resolved": []}
        if self.unsolved and self._dirty_since_retry:
            self._dirty_since_retry = False
            for tid, pairs in list(self.unsolved.items()):
                res = self.attempt(tid, pairs, effort=effort)
                if res["solved"]:
                    report["resolved"].append((tid, str(res["program"]), res["checked"]))
        for wid, spec in list(self.unsolved_worlds.items()):     # думать дольше = исследовать дольше
            res = self.explore(wid, spec, effort=len(WORLD_LADDER))
            if res["solved"]:
                report["worlds_resolved"].append((wid, res["steps"], res["optimal"],
                                                  res["explored"]))
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
                 "world_maps": self.world_maps,
                 "unsolved_worlds": self.unsolved_worlds,
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
        from thinking_system.language.morphology import stem

        self.lexicon.words.update(state["lexicon"])
        for w in state["lexicon"]:                           # восстановить индекс основ
            self.lexicon._stems[stem(w)] = w
        self.solutions.update(state["solutions"])
        self.unsolved.update(state["unsolved"])
        self.world_maps.update(state.get("world_maps", {}))
        self.unsolved_worlds.update(state.get("unsolved_worlds", {}))
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
