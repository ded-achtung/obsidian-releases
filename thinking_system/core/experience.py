"""Единица накопленного опыта.

Один Experience — это пара «контекст → цель» в ЛАТЕНТНОМ пространстве:
что система видела (context) и что за этим последовало (target). Именно такие
пары копятся в эпизодической памяти и переигрываются (replay) при обучении
предиктора. Это минимальный «след опыта», из которого со временем рождается
модель мира.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Experience:
    """Эпизод: контекст (один или несколько прошлых латентов) и целевой латент.

    Attributes:
        context: латентный контекст, форма (context_len * latent_dim,).
        target:  целевой латент, который нужно предсказать, форма (latent_dim,).
        t:       номер шага во времени (для отладки/анализа дрейфа).
    """

    context: np.ndarray
    target: np.ndarray
    t: int
