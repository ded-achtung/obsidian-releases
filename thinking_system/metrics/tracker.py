"""Сбор и сводка метрик цикла.

Главная метрика — ошибка предсказания латента во времени. «Понимание растёт»,
если скользящая ошибка падает И опускается ниже тривиального baseline
(persistence: «следующее ≈ предыдущее»). Также считаем «sample-efficiency»-прокси:
сколько шагов нужно, чтобы вдвое сократить начальную ошибку.

Это MVP-набор; по дорожной карге сюда добавятся continual-learning метрики
(forward/backward transfer, forgetting) при переходе к нескольким задачам/доменам.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MetricsTracker:
    """Накапливает по-шаговые метрики и считает сводку.

    Args:
        window: окно скользящего среднего для ошибки/baseline.
    """

    window: int = 200
    steps: list[int] = field(default_factory=list)
    errors: list[float] = field(default_factory=list)
    baselines: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)

    def record(self, *, t: int, error: float, baseline: float, loss: float | None) -> None:
        self.steps.append(t)
        self.errors.append(error)
        self.baselines.append(baseline)
        if loss is not None:
            self.losses.append(loss)

    def rolling(self, values: list[float] | None = None) -> np.ndarray:
        """Скользящее среднее (по умолчанию — по ошибке предсказания)."""
        arr = np.asarray(self.errors if values is None else values, dtype=np.float64)
        if arr.size == 0:
            return arr
        w = min(self.window, arr.size)
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode="valid")

    def summary(self) -> dict[str, float]:
        """Сводка: начальная/финальная ошибка, baseline, улучшение, sample-efficiency."""
        if len(self.errors) < 2:
            return {"n": float(len(self.errors))}

        roll = self.rolling()
        roll_base = self.rolling(self.baselines)
        seg = max(1, roll.size // 10)  # усредняем по первым/последним 10 %

        initial = float(np.mean(roll[:seg]))
        final = float(np.mean(roll[-seg:]))
        baseline_final = float(np.mean(roll_base[-seg:]))

        # Шаги до 50 %-сокращения начальной ошибки (прокси sample-efficiency).
        target = initial * 0.5
        below = np.where(roll <= target)[0]
        steps_to_halve = float(below[0]) if below.size else float("nan")

        return {
            "n": float(len(self.errors)),
            "initial_error": initial,
            "final_error": final,
            "baseline_final": baseline_final,
            "improvement_pct": float(100.0 * (initial - final) / initial) if initial > 0 else 0.0,
            "beats_baseline": float(final < baseline_final),
            "steps_to_halve_error": steps_to_halve,
        }
