"""Активный вывод: выбор действия по ожидаемой свободной энергии (MVP).

Ожидаемая свободная энергия действия:

    G(a) = − [ epistemic(a) + pragmatic(a) ]

Агент выбирает действия, минимизирующие G, то есть максимизирующие сумму
эпистемической (прирост информации / любопытство) и прагматической (достижение
целей) ценности. Выбор — стохастический через softmax с точностью γ:

    P(a) ∝ exp( γ · [epistemic(a) + pragmatic(a)] )

В MVP прагматическая ценность = 0 (целей пока нет) — остаётся чистое любопытство.
Эпистемическую ценность даёт цикл (learning-progress прокси прироста информации).

СВЯЗЬ С pymdp: это та же идея EFE, что в pymdp/RxInfer для дискретных POMDP, но
посчитанная для нашей непрерывной модели мира. Когда понадобится точная
дискретная формулировка (матрицы A/B/C/D, прагматика через предпочтения C) —
сюда встаёт pymdp-политика с тем же интерфейсом Policy.
"""

from __future__ import annotations

import numpy as np

from thinking_system.policies.base import Policy


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class ActiveInferencePolicy(Policy):
    """Стохастический выбор действия по ожидаемой свободной энергии.

    Args:
        gamma: точность (precision) — чем выше, тем решительнее выбор лучшего действия.
        seed: зерно ГСЧ сэмплирования.
    """

    def __init__(self, *, gamma: float = 8.0, seed: int = 0) -> None:
        self.gamma = gamma
        self._rng = np.random.default_rng(seed)

    def select_action(self, epistemic: np.ndarray, pragmatic: np.ndarray | None = None) -> int:
        epi = np.asarray(epistemic, dtype=np.float64)
        prag = np.zeros_like(epi) if pragmatic is None else np.asarray(pragmatic, dtype=np.float64)
        neg_free_energy = self.gamma * (epi + prag)  # = −γ·G
        probs = _softmax(neg_free_energy)
        return int(self._rng.choice(len(probs), p=probs))


class RandomPolicy(Policy):
    """Равномерный случайный выбор — baseline «внимание без любопытства»."""

    def __init__(self, n_actions: int, *, seed: int = 0) -> None:
        self.n_actions = n_actions
        self._rng = np.random.default_rng(seed)

    def select_action(self, epistemic: np.ndarray | None = None, pragmatic: np.ndarray | None = None) -> int:
        return int(self._rng.integers(self.n_actions))
