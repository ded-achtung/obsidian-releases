"""Метрики непрерывного обучения (по Lopez-Paz & Ranzato, GEM, arXiv:1706.08840).

Оценщик держит ФИКСИРОВАННЫЕ eval-наборы по каждой задаче и после каждой
обученной задачи замеряет ошибку предсказания по ВСЕМ задачам, заполняя матрицу
E[i][j] = ошибка на задаче j после обучения по задачу i. Из неё:

  • ACC        — средняя финальная ошибка по задачам (меньше — лучше);
  • forgetting — среднее РОСТА ошибки на старых задачах (финал − сразу-после-обучения);
                 ~0 = не забывает, большое = катастрофическое забывание;
  • BWT        — backward transfer (в терминах «выше=лучше»): отрицателен при забывании;
  • FWT        — forward transfer: помогли ли прошлые задачи новой (ошибка до её
                 обучения vs случайная инициализация).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from thinking_system.encoders.base import Encoder
from thinking_system.predictors.base import Predictor
from thinking_system.tasks.sequence import Task


class ContinualEvaluator:
    """Заполняет матрицу задача×задача и считает ACC/forgetting/BWT/FWT.

    Args:
        encoder: общий энкодер (тот же, что в обучении).
        tasks: список задач.
        context_len: длина контекста (как в обучении).
        n_eval: число фиксированных eval-пар на задачу.
    """

    def __init__(self, encoder: Encoder, tasks: list[Task], *, context_len: int = 2, n_eval: int = 256) -> None:
        self.context_len = context_len
        self.task_names = [t.name for t in tasks]
        self._eval = [self._build_eval(encoder, t, n_eval) for t in tasks]
        self._after: list[int] = []
        self._rows: list[np.ndarray] = []

    def _build_eval(self, encoder: Encoder, task: Task, n_eval: int) -> tuple[np.ndarray, np.ndarray]:
        history: deque[np.ndarray] = deque(maxlen=self.context_len + 1)
        contexts: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for obs in task.stream():
            history.append(encoder.encode(np.asarray(obs, dtype=np.float64)))
            if len(history) == self.context_len + 1:
                contexts.append(np.concatenate(list(history)[:-1]))
                targets.append(history[-1])
                if len(contexts) >= n_eval:
                    break
        return np.array(contexts), np.array(targets)

    def eval_errors(self, predictor: Predictor) -> np.ndarray:
        """Средняя ошибка предсказания по каждой задаче на её фиксированном наборе."""
        errs = []
        for contexts, targets in self._eval:
            preds = np.array([predictor.predict(c) for c in contexts])
            errs.append(float(np.mean(np.sum((preds - targets) ** 2, axis=1))))
        return np.array(errs)

    def record(self, after_task: int, predictor: Predictor) -> None:
        self._after.append(after_task)
        self._rows.append(self.eval_errors(predictor))

    def matrix(self) -> np.ndarray:
        """Матрица ошибок: строка 0 = случайная инициализация, далее после задачи 0,1,…"""
        return np.array(self._rows)

    def metrics(self) -> dict[str, float]:
        m = self.matrix()
        k = len(self.task_names)
        rand = m[0]                                       # случайная инициализация
        final = m[-1]                                     # после последней задачи
        diag = np.array([m[i + 1][i] for i in range(k)])  # ошибка на задаче i сразу после неё

        acc = float(final.mean())
        forgetting = float(np.mean(final[: k - 1] - diag[: k - 1])) if k > 1 else 0.0
        bwt = float(np.mean(diag[: k - 1] - final[: k - 1])) if k > 1 else 0.0  # >0 хуже инвертируется
        pre = np.array([m[j][j] for j in range(1, k)])    # ошибка на задаче j до её обучения
        fwt = float(np.mean(rand[1:] - pre)) if k > 1 else 0.0  # >0 = прошлое помогло

        return {"acc": acc, "forgetting": forgetting, "bwt": bwt, "fwt": fwt}
