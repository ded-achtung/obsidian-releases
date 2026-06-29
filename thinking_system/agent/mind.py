"""MindAgent — единый контур: общая память + маршрутизация задачи по модулям.

Разрыв «нет единого я»: способности жили в отдельных демо. Здесь ОДИН агент с общей
доской (blackboard) принимает задачу и САМ маршрутизирует её по модулям, связывая их
в цепочку:

    ВОСПРИЯТИЕ/РАССУЖДЕНИЕ — вывести динамику мира из наблюдений (induction);
    СМЫСЛ                  — понять языковую команду (подсловные эмбеддинги, #смысл);
    ПЛАНИРОВАНИЕ           — построить путь к понятой цели по выученной модели;
    ДЕДУКЦИЯ              — ответить на факт-вопрос о мире (relational KB).

Композитная задача (язык→цель + рассуждённая модель + план) не решается одним
модулем — нужна их сцепка через общую память. Честно: это ОРКЕСТРАЦИЯ реальных
модулей под единой памятью и маршрутизатором, а не возникшее само-собой мышление;
но каждый модуль здесь нагружен (см. абляцию: без смысла композит рушится).
"""

from __future__ import annotations

from thinking_system.world.gridworld import GridWorld
from thinking_system.reasoning.grounded import observe, induce_dynamics, WorldRule
from thinking_system.language.grounding import (
    BagOfWords, CharNgram, GoalClassifier, generate_commands, goal_cell,
)
from thinking_system.language.facts import FactReader


class MindAgent:
    """Один агент: общая память (blackboard) + маршрутизация по модулям + сцепка."""

    def __init__(self, grid: GridWorld, *, semantic: bool = True, seed: int = 0) -> None:
        self.g = grid
        self.seed = seed
        self.bb: dict = {}                      # общая память: сюда складываются результаты модулей
        self.trace: list[str] = []             # какой модуль отработал (прозрачность)

        # МОДУЛЬ СМЫСЛА: подсловный (обобщает на новые словоформы) или baseline-мешок слов
        cmds = generate_commands()
        feat = CharNgram([t for t, _ in cmds]) if semantic else BagOfWords([t for t, _ in cmds])
        self.goal_clf = GoalClassifier(feat)
        self.goal_clf.fit(cmds, epochs=400)
        self.semantic = semantic

    # ── МОДУЛЬ РАССУЖДЕНИЯ: вывести динамику мира из наблюдений ───────────────────
    def perceive_dynamics(self, *, start_steps: int = 20, cap: int = 200) -> int:
        steps = start_steps
        rules = induce_dynamics(observe(self.g, steps, seed=self.seed))
        while len(rules) < 4 and steps < cap:
            steps += 10
            rules = induce_dynamics(observe(self.g, steps, seed=self.seed))
        self.bb["world_model"] = WorldRule(self.g, rules)
        self.bb["observations"] = steps
        self.trace.append(f"reasoning:{len(rules)}rules/{steps}obs")
        return len(rules)

    # ── МОДУЛЬ СМЫСЛА: понять команду → клетка-цель ──────────────────────────────
    def understand(self, command: str) -> tuple[int, int]:
        cls, conf = self.goal_clf.predict(command)
        cell = goal_cell(cls, self.g.size)
        self.bb["goal"] = cell
        self.bb["goal_cls"] = cls
        self.trace.append(f"meaning:cls={cls}")
        return cell

    # ── МОДУЛЬ ПЛАНИРОВАНИЯ: путь к цели по выученной модели + исполнение ─────────
    def plan_and_act(self, start: tuple[int, int]) -> dict:
        model: WorldRule = self.bb["world_model"]
        goal = self.bb["goal"]
        path = model.plan(start, goal)
        s = start
        if path is not None:
            for a in path:
                s = model.predict(s, a)
        self.trace.append(f"planning:{'ok' if path else 'none'}")
        return {"reached": s == goal and path is not None, "steps": len(path) if path else None, "goal": goal}

    # ── МОДУЛЬ ДЕДУКЦИИ: ответить на факт-вопрос (другой тип рассуждения) ─────────
    def ask(self, statements: list[str], question: str) -> str:
        fr = FactReader()
        for st in statements:
            fr.tell(st)
        self.trace.append("deduction")
        return fr.answer(question)

    # ── РОУТЕР: композитная задача = цепочка модулей через общую память ───────────
    def solve_navigation(self, command: str, start: tuple[int, int]) -> dict:
        """Композит: рассуждение(динамика) → смысл(команда) → план(путь) → действие."""
        self.trace = []
        if "world_model" not in self.bb:
            self.perceive_dynamics()
        self.understand(command)
        return self.plan_and_act(start)
