"""ThinkingSystem — ЕДИНАЯ думающая система: один агент, одна память, общий язык.

Раньше способности жили порознь (каждый run_*.py собирал свои объекты под одну
задачу). Здесь они — ФАКУЛЬТЕТЫ ОДНОГО объекта с ОБЩИМ состоянием, и состояние
ПЕРЕТЕКАЕТ между ними (то, что делает это «одной системой», а не фасадом над
демками). Архитектура — по теории Complementary Learning Systems: три памяти
одного агента + переносы между ними.

  • ПРОЦЕДУРНАЯ память — ОДНА `Library`, которую делят язык (GroundedLexicon) и
    рассуждение (induce). Поэтому навык, ВЫУЧЕННЫЙ ЧТЕНИЕМ, доступен РАССУЖДЕНИЮ;
    а программа, НАЙДЕННАЯ ПОИСКОМ, после `name_skill` становится СЛОВОМ языка и
    примитивом библиотеки (поиск на будущее короче). Это и есть рост словаря.
  • СЕМАНТИЧЕСКАЯ память — ОДНИ `FactReader` (KB) и `CausalReader`. Чтение пишет в
    них факты и причины; вопросы отвечаются ДЕДУКЦИЕЙ по тому же хранилищу.
  • ЭПИЗОДИЧЕСКАЯ память + ПРЕДСКАЗАНИЕ + ЛЮБОПЫТСТВО — общий «кортекс»
    (CognitiveAgent): тот же сигнал ошибки предсказания и учит модель, и направляет
    внимание, и ловит аномалии на байтовых потоках.
  • ДЕЙСТВИЕ в мире — та же связка предсказание+активный вывод+память переносится
    на навигацию (ActingAgent), цель грунтится тем же языком.

Единый интерфейс: read · ask · solve · name_skill · understand · perceive · act ·
mind. Всё накапливается в ОДНОМ объекте и переживает между вызовами.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from thinking_system.agent.acting import ActingAgent
from thinking_system.agent.cognitive import CognitiveAgent
from thinking_system.language.causal_lang import CausalReader, _MARK
from thinking_system.language.facts import FactReader
from thinking_system.language.reader import LessonReader
from thinking_system.language.understanding import GroundedLexicon, tokenize
from thinking_system.reasoning.induction import Library, Program, default_primitives
from thinking_system.reasoning.invention import invent
from thinking_system.reasoning.predicates import induce_predicate
from thinking_system.text.vocab import ByteVocab
from thinking_system.world.gridworld import default_maze


class ThinkingSystem:
    """Единый думающий агент: общая память + общий язык + переносы между факультетами."""

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = seed
        # ── общая ПРОЦЕДУРНАЯ память: одна библиотека делится языком и рассуждением ──
        self.library = Library(default_primitives())
        self.lexicon = GroundedLexicon(library=self.library)
        # ── общая СЕМАНТИЧЕСКАЯ память: факты и причины ──
        self.facts = FactReader()
        self.causal = CausalReader()
        # маршрутизатор чтения пишет в ОБЩИЕ компоненты выше
        self.reader = LessonReader(lex=self.lexicon, facts=self.facts, causal=self.causal)
        # ── эпизодическая/предиктивная кора (ленивая) и мир ──
        self.cortex: CognitiveAgent | None = None
        self.vocab = ByteVocab()
        self.world_env = None
        self.world_agent: ActingAgent | None = None
        self._lib_names: set[str] = {p.name for p in self.library.prims}
        self.episode: list[str] = []  # автобиографический след: что система делала

    # ── ПРОЦЕДУРНОЕ: держать библиотеку и словарь синхронными ──────────────────────
    def _sync_library(self) -> list[str]:
        """Зарегистрировать новые слова словаря как примитивы библиотеки (рост языка).

        После этого ЧТЕНИЕ обогащает поиск РАССУЖДЕНИЯ: выученное словом комбо
        становится одношаговым макросом, и будущие задачи решаются меньшей глубиной.
        """
        added = []
        for word, prog in self.lexicon.words.items():
            if word not in self._lib_names:
                self.library.add_abstraction(word, prog)
                self._lib_names.add(word)
                added.append(word)
        return added

    # ── ЧТЕНИЕ: текст → слова + факты + причины в ОБЩУЮ память ──────────────────────
    def read(self, text: str) -> dict:
        """Прочитать текст: выучить слова-операции, факты и причины в общую память."""
        before = dict(self.reader.stats)
        stats = self.reader.read(text)
        grown = self._sync_library()  # рост процедурного словаря из прочитанного
        delta = {k: stats[k] - before.get(k, 0) for k in stats}
        self.episode.append(f"read: +{delta['слова']} слов, +{delta['факты']} фактов, "
                            f"+{delta['причины']} причин; библиотека +{len(grown)}")
        return {"stats": stats, "learned_words": grown}

    # ── ВОПРОС: дедукция по общей семантической памяти (факты или причины) ──────────
    def ask(self, question: str) -> str:
        """Ответить ДЕДУКЦИЕЙ: причинный вопрос → causal, иначе → факты."""
        toks = tokenize(question)
        if "если" in toks or any(m in toks for m in _MARK):
            ans = self.causal.answer(question)
        else:
            ans = self.facts.answer(question)
        self.episode.append(f"ask: {question!r} → {ans}")
        return ans

    # ── ПОНИМАНИЕ: разобрать предложение в слова (читанные/выученные) и ИСПОЛНИТЬ ────
    def understand(self, sentence: str) -> dict:
        """Понять предложение как композицию известных слов и выполнить его."""
        res = self.lexicon.solve(sentence)
        self.episode.append(f"understand: {sentence!r} → {res.get('answer', res)}")
        return res

    # ── РАССУЖДЕНИЕ: синтез программы из примеров по ОБЩЕЙ библиотеке ────────────────
    def solve(self, examples: list[tuple], *, max_depth: int = 3, invent: bool = True) -> Program | None:
        """Вывести программу из 2-3 примеров: сперва КОМПОЗИЦИЯ известного (Оккам),

        затем — если не вышло и invent=True — ИЗОБРЕТЕНИЕ операции из наблюдаемой
        регулярности (новая операция строится из того, что видно в данных, и
        добавляется в библиотеку для переиспользования).
        """
        prog = self.library.induce(examples, max_depth=max_depth)
        if prog is None and invent:
            prim = self.invent_operation(examples)
            if prim is not None:
                prog = Program([prim])
        self.episode.append(f"solve: {len(examples)} примеров → {prog}")
        return prog

    # ── ИЗОБРЕТЕНИЕ: новая операция ИЗ НАБЛЮДЕНИЙ (не из воздуха) → в библиотеку ─────
    def invent_operation(self, examples: list[tuple]) -> "Primitive | None":  # type: ignore[name-defined]
        """Подогнать грунтованный шаблон к примерам; родившуюся операцию — в библиотеку.

        Сначала прямолинейная регулярность (аффинная/квадратичная/поэлементная),
        затем — УСЛОВНАЯ операция «если P(x): f иначе g», тоже выведенная из данных.
        """
        prim = invent(examples)
        if prim is None:
            self.episode.append(f"invent: из {len(examples)} примеров регулярность не найдена")
            return None
        if prim.name not in self._lib_names:           # рост библиотеки из наблюдаемого
            self.library.prims.append(prim)
            self.library.abstractions.append(prim.name)
            self._lib_names.add(prim.name)
        self.episode.append(f"invent: новая операция «{prim.name}» из наблюдений")
        return prim

    # ── ИНДУКЦИЯ ПРЕДИКАТА: выучить УСЛОВИЕ из размеченных примеров (не хардкод) ─────
    def learn_predicate(self, labeled: list[tuple]) -> str | None:
        """Вывести предикат «вход → да/нет» из примеров; вернуть его имя (или None)."""
        res = induce_predicate(labeled)
        name = res[0] if res is not None else None
        self.episode.append(f"learn_predicate: {len(labeled)} примеров → {name}")
        return name

    # ── ПЕРЕНОС рассуждение→язык: назвать найденный навык (станет словом и примитивом) ─
    def name_skill(self, word: str, examples: list[tuple], *, max_depth: int = 3) -> bool:
        """Найти навык поиском и НАЗВАТЬ его: слово языка + примитив библиотеки.

        Это замыкает контур роста: способность, открытая РАССУЖДЕНИЕМ, становится
        доступна ЯЗЫКУ (understand «слово …») и сокращает будущий поиск (глубина↓).
        """
        prog = self.library.induce(examples, max_depth=max_depth)
        if prog is None:                               # не вышло композицией — изобрести из данных
            prim = invent(examples)                    # прямолинейная ИЛИ условная (ветвящаяся)
            prog = Program([prim]) if prim is not None else None
        if prog is None:
            self.episode.append(f"name_skill: {word!r} — не выведено")
            return False
        key = self.lexicon.normalize(word)
        self.library.add_abstraction(key, prog)       # рассуждение: новый примитив
        self._lib_names.add(key)
        self.lexicon.words[key] = prog                # язык: новое слово
        self.episode.append(f"name_skill: {word!r} = «{prog}» (язык+библиотека)")
        return True

    # ── ВОСПРИЯТИЕ потоков: предсказание + любопытство + эпизодическая память ───────
    def perceive(self, sources: list[tuple[str, str]], *, steps: int = 4000) -> dict:
        """Читать байтовые потоки кортексом: surprise учит модель и ведёт внимание.

        sources: список (имя, текст). Возвращает падение bits-per-byte по источникам.
        """
        encoded = [(name, self.vocab.encode(text)) for name, text in sources]
        if self.cortex is None:
            self.cortex = CognitiveAgent(encoded, seed=self.seed)
        before = [self.cortex.source_bpc(k) for k in range(self.cortex.K)]
        self.cortex.run(steps)
        after = [self.cortex.source_bpc(k) for k in range(self.cortex.K)]
        self.episode.append(f"perceive: {steps} шагов, bpc {np.mean(before):.2f}→{np.mean(after):.2f}")
        return {"names": self.cortex.names, "bpc_before": before, "bpc_after": after,
                "attention": (self.cortex.visits / max(1, self.cortex.visits.sum())).tolist()}

    def scan(self, text: str) -> np.ndarray:
        """Просканировать новый текст обученным кортексом: surprise по байтам (аномалии)."""
        if self.cortex is None:
            raise RuntimeError("кортекс пуст: сначала вызовите perceive(...)")
        return self.cortex.scan(text)

    # ── ДЕЙСТВИЕ в мире: модель мира + активный вывод + память (та же связка) ───────
    def learn_world(self, env=None, *, episodes: int = 40, max_steps: int = 1500) -> dict:
        """Освоить незнакомый мир: исследование→карта→достижение цели всё быстрее."""
        self.world_env = env if env is not None else default_maze()
        self.world_agent = ActingAgent(self.world_env.n_actions, self.world_env.goal_state, seed=self.seed)
        hist = []
        for ep in range(episodes):
            eps = max(0.05, 0.5 * (0.88 ** ep))
            s = self.world_env.reset()
            for t in range(1, max_steps + 1):
                a = self.world_agent.act(s, epsilon=eps)
                sp, done = self.world_env.step(a)
                self.world_agent.learn(s, a, sp)
                s = sp
                if done:
                    break
            hist.append(t)
        self.episode.append(f"learn_world: {hist[0]}→{hist[-1]} шагов (оптимум {self.world_env.optimal_steps()})")
        return {"steps_first": hist[0], "steps_last": hist[-1], "optimal": self.world_env.optimal_steps(), "history": hist}

    def act(self) -> dict:
        """Пройти к цели по ВЫУЧЕННОЙ карте мира."""
        if self.world_agent is None or self.world_env is None:
            raise RuntimeError("мир не освоен: сначала вызовите learn_world(...)")
        path = self.world_agent.greedy_path(self.world_env.reset())
        reached = bool(path and path[-1] == self.world_env.goal_state)
        self.episode.append(f"act: путь {len(path) - 1} шагов, дошёл={reached}")
        return {"path": path, "steps": len(path) - 1, "reached": reached}

    # ── ЕДИНОЕ СОСТОЯНИЕ: всё нажитое — в одном объекте ────────────────────────────
    def mind(self) -> dict:
        """Снимок «состояния ума»: всё, что система нажила всеми факультетами."""
        return {
            "слова (язык)": sorted(self.lexicon.words),
            "абстракции (библиотека)": list(self.library.abstractions),
            "фактов (семантика)": len(self.facts.kb.facts),
            "выводимых фактов (замыкание)": len(self.facts.kb.closure()),
            "причинных связей": {k: list(v) for k, v in self.causal.parents.items()},
            "эпизодическая память (байт)": len(self.cortex.memory) if self.cortex else 0,
            "мир освоен": self.world_env is not None,
        }
