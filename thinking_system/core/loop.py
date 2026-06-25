"""PredictiveLoop — замкнутый цикл «предсказание → ошибка → обучение».

Это сердце MVP. На каждом шаге потока:

  1. наблюдение  obs_t        → энкодер → латент z_t
  2. контекст    = последние context_len латентов
  3. предсказание ẑ_t          = predictor(context)
  4. ошибка       = ||ẑ_t - z_t||²   (сигнал обучения; метрика «понимания»)
  5. опыт (context → z_t) кладётся в эпизодическую память
  6. периодически: replay-батч из памяти дообучает предиктор

Со временем ошибка предсказания падает — система всё лучше «понимает»
временную структуру потока. Для сравнения считается тривиальный baseline
(persistence: «следующее = предыдущее»), который обучаемый предиктор должен
обойти.
"""

from __future__ import annotations

from collections import deque
from itertools import islice

import numpy as np

from thinking_system.core.experience import Experience
from thinking_system.encoders.base import Encoder
from thinking_system.memory.base import EpisodicMemory
from thinking_system.metrics.tracker import MetricsTracker
from thinking_system.predictors.base import Predictor
from thinking_system.streams.base import Stream


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.dot(diff, diff))


class PredictiveLoop:
    """Оркестратор цикла предсказание-ошибка-обучение на одном потоке.

    Args:
        encoder: сенсорный энкодер obs → латент.
        predictor: предиктор латент-контекст → следующий латент (обучаемый).
        memory: эпизодическая память (хранит опыт, отдаёт replay-батчи).
        context_len: сколько прошлых латентов образуют контекст.
        train_every: как часто (в шагах) запускать обучение по replay-батчу.
        batch_size: размер replay-батча.
        warmup: число шагов до старта обучения (накопить опыт в памяти).
        tracker: сборщик метрик (создаётся автоматически, если не передан).
    """

    def __init__(
        self,
        encoder: Encoder,
        predictor: Predictor,
        memory: EpisodicMemory,
        *,
        context_len: int = 2,
        train_every: int = 4,
        batch_size: int = 64,
        warmup: int = 128,
        tracker: MetricsTracker | None = None,
    ) -> None:
        self.encoder = encoder
        self.predictor = predictor
        self.memory = memory
        self.context_len = context_len
        self.train_every = train_every
        self.batch_size = batch_size
        self.warmup = warmup
        self.tracker = tracker if tracker is not None else MetricsTracker()

    def run(self, stream: Stream, n_steps: int) -> MetricsTracker:
        """Прогнать цикл на n_steps шагах потока и вернуть собранные метрики."""
        history: deque[np.ndarray] = deque(maxlen=self.context_len + 1)

        for t, obs in enumerate(islice(iter(stream), n_steps)):
            z = self.encoder.encode(np.asarray(obs, dtype=np.float64))
            history.append(z)

            # Нужен полный контекст (context_len латентов) + цель (текущий латент).
            if len(history) < self.context_len + 1:
                continue

            context = np.concatenate(list(history)[:-1])  # context_len прошлых латентов
            target = history[-1]                          # что реально случилось

            predicted = self.predictor.predict(context)
            err = _mse(predicted, target)
            baseline = _mse(history[-2], target)          # persistence: next ≈ prev

            self.memory.add(Experience(context.copy(), target.copy(), t))

            loss: float | None = None
            if t >= self.warmup and t % self.train_every == 0 and len(self.memory) >= self.batch_size:
                contexts, targets = self.memory.sample(self.batch_size)
                loss = self.predictor.update(contexts, targets)

            self.tracker.record(t=t, error=err, baseline=baseline, loss=loss)

        return self.tracker
