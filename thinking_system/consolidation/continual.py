"""ContinualLearner — обучение по последовательности задач с consolidation/replay.

Сравнивает два режима на одной общей модели мира:

  • replay=True  (CLS-консолидация): эпизодическая память сохраняется МЕЖДУ задачами;
    каждый шаг обучения берёт replay-батч из ПОЛНОЙ памяти (старое + новое) →
    старые задачи переигрываются (как консолидация во сне) → не забываются.
  • replay=False (наивно): память сбрасывается в начале каждой задачи → обучение
    идёт только на текущей задаче → старые затираются (катастрофическое забывание).

Разница ТОЛЬКО в сохранении/сбросе памяти — это изолирует эффект повторения.
Энкодер и предиктор общие; контекст БЕЗ id задачи (иначе забывания бы не было —
сеть просто разнесла бы задачи). После каждой задачи зовётся evaluator, который
заполняет строку матрицы «ошибка по всем задачам».
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from itertools import islice

import numpy as np

from thinking_system.core.experience import Experience
from thinking_system.encoders.base import Encoder
from thinking_system.memory.base import EpisodicMemory
from thinking_system.predictors.base import Predictor
from thinking_system.tasks.sequence import Task


class ContinualLearner:
    """Непрерывное обучение по списку задач с режимом replay/наивно.

    Args:
        encoder: общий энкодер obs → латент.
        predictor: общая модель мира (обучается через все задачи).
        memory_factory: callable → свежая EpisodicMemory (для сброса в наивном режиме).
        replay: True — память живёт между задачами (CLS); False — сброс на каждой задаче.
        context_len, train_every, batch_size, warmup: как в предиктивном цикле.
    """

    def __init__(
        self,
        encoder: Encoder,
        predictor: Predictor,
        memory_factory: Callable[[], EpisodicMemory],
        *,
        replay: bool = True,
        context_len: int = 2,
        train_every: int = 4,
        batch_size: int = 64,
        warmup: int = 64,
    ) -> None:
        self.encoder = encoder
        self.predictor = predictor
        self.memory_factory = memory_factory
        self.memory = memory_factory()
        self.replay = replay
        self.context_len = context_len
        self.train_every = train_every
        self.batch_size = batch_size
        self.warmup = warmup

    def _train_task(self, stream, steps: int) -> None:
        history: deque[np.ndarray] = deque(maxlen=self.context_len + 1)
        for t, obs in enumerate(islice(iter(stream), steps)):
            z = self.encoder.encode(np.asarray(obs, dtype=np.float64))
            history.append(z)
            if len(history) < self.context_len + 1:
                continue
            context = np.concatenate(list(history)[:-1])  # БЕЗ id задачи
            target = history[-1]
            self.memory.add(Experience(context.copy(), target.copy(), t))
            if t >= self.warmup and t % self.train_every == 0 and len(self.memory) >= self.batch_size:
                contexts, targets = self.memory.sample(self.batch_size)
                self.predictor.update(contexts, targets)

    def run(self, tasks: list[Task], steps_per_task: int, evaluator) -> object:
        """Пройти задачи по очереди, после каждой — оценить все задачи."""
        evaluator.record(-1, self.predictor)  # строка для случайной инициализации (для FWT)
        for ti, task in enumerate(tasks):
            if not self.replay:
                self.memory = self.memory_factory()  # наивно: забыть прошлый опыт
            self._train_task(task.stream(), steps_per_task)
            evaluator.record(ti, self.predictor)
        return evaluator
