"""ActiveInferenceLoop — предиктивный цикл + активный выбор наблюдений (шаг 2).

Расширяет пассивный PredictiveLoop: агент САМ выбирает, какой канал наблюдать,
по эпистемической ценности (любопытству). Ценность канала = СТАТИСТИЧЕСКИ ЗНАЧИМЫЙ
learning progress: устойчивое падение ошибки предсказания на длинном окне.

Почему именно так (подтверждено замерами на нашей среде):
  • выучиваемый, ещё не выученный канал (напр. «hard») даёт устойчивое падение
    ошибки при низкой дисперсии → высокий z → высокая ценность; внимание держится,
    пока есть прогресс, и уходит после освоения — естественный куррикулум;
  • шумный канал имеет огромную дисперсию ошибки и нулевой реальный тренд →
    z мал → ценность 0. Это снимает проблему «шумного телевизора», на которой
    залипают и наивная ошибка, и рассогласование ансамбля (оба тянут на шум).

Окно длинное (ловит медленный прогресс трудного канала); z-гейт оценивает
значимость падения относительно дисперсии канала. После гейта ценности
нормируются по максимуму — когда есть один ещё-выучиваемый канал, он получает
почти всё внимание; когда учиться нечему, все ценности 0 → равномерно (корректно).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from thinking_system.core.experience import Experience
from thinking_system.encoders.base import Encoder
from thinking_system.envs.base import Environment
from thinking_system.memory.base import EpisodicMemory
from thinking_system.metrics.curiosity import CuriosityTracker
from thinking_system.policies.base import Policy
from thinking_system.predictors.base import Predictor


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.dot(diff, diff))


class ActiveInferenceLoop:
    """Цикл «выбрать наблюдение → предсказать → ошибка → обучение» с любопытством.

    Args:
        encoder, predictor, memory: как в PredictiveLoop (predictor.context_dim
            включает one-hot канала: context_len*latent_dim + n_actions).
        policy: политика выбора действия (ActiveInferencePolicy / RandomPolicy).
        n_actions: число каналов/действий среды.
        context_len, train_every, batch_size, warmup: как в шаге 1.
        lp_window: длина окна ошибок для оценки learning progress (длинное — чтобы
            ловить медленный прогресс трудных каналов).
        lp_min_data: минимум ошибок до доверия LP (иначе epi=0 → равномерная разведка).
        lp_floor: абсолютный минимум значимого падения ошибки.
        lp_z: порог значимости падения (z-оценка) — гасит шумные каналы.
        tracker: сборщик метрик любопытства.
    """

    def __init__(
        self,
        encoder: Encoder,
        predictor: Predictor,
        memory: EpisodicMemory,
        policy: Policy,
        n_actions: int,
        *,
        context_len: int = 2,
        train_every: int = 4,
        batch_size: int = 64,
        warmup: int = 128,
        lp_window: int = 1000,
        lp_min_data: int = 500,
        lp_floor: float = 0.005,
        lp_z: float = 2.5,
        tracker: CuriosityTracker | None = None,
    ) -> None:
        self.encoder = encoder
        self.predictor = predictor
        self.memory = memory
        self.policy = policy
        self.K = n_actions
        self.context_len = context_len
        self.train_every = train_every
        self.batch_size = batch_size
        self.warmup = warmup
        self.lp_window = lp_window
        self.lp_min_data = lp_min_data
        self.lp_floor = lp_floor
        self.lp_z = lp_z
        self.tracker = tracker if tracker is not None else CuriosityTracker(n_actions=n_actions)

        self._history: list[deque[np.ndarray]] = [deque(maxlen=context_len + 1) for _ in range(n_actions)]
        self._visits = np.zeros(n_actions, dtype=int)
        self._err_hist: list[deque[float]] = [deque(maxlen=lp_window) for _ in range(n_actions)]

    def _onehot(self, k: int) -> np.ndarray:
        v = np.zeros(self.K)
        v[k] = 1.0
        return v

    def epistemic_values(self) -> np.ndarray:
        """Эпистемическая ценность по каналам = learning progress, прошедший гейт (нормир.).

        Гейт сравнивает старую и новую половины окна ошибок: падение должно быть
        и абсолютно заметным (≥ lp_floor), и большим относительно ДИСПЕРСИИ ошибок
        канала (drop/SE ≥ lp_z). Высокая дисперсия шумного канала → большой SE → гейт
        его глушит (так снимается «шумный телевизор» с БЕЛЫМ шумом).

        ЧЕСТНО О ГРАНИЦАХ: это ЭВРИСТИЧЕСКИЙ гейт, а НЕ строгий тест значимости. Ошибки —
        автокоррелированный ряд, поэтому SE по сырым выборкам занижен; гейт надёжно
        отсекает каналы с высокодисперсным белым шумом (как в нашей среде), но низко-
        дисперсный автокоррелированный ДРЕЙФ в принципе может его обмануть. Для строгости
        нужен тренд-тест с поправкой на автокорреляцию (Newey–West / Mann–Kendall).
        """
        epi = np.zeros(self.K)
        for k in range(self.K):
            hist = self._err_hist[k]
            if len(hist) < self.lp_min_data:
                continue  # данных мало → epi=0; softmax даст равномерную разведку
            arr = np.fromiter(hist, dtype=np.float64)
            half = arr.size // 2
            old, new = arr[:half], arr[arr.size - half:]  # симметричные половины (корректно при нечётном размере)
            drop = float(old.mean() - new.mean())  # падение ошибки = прогресс
            if drop < self.lp_floor:
                continue
            # SE разности средних по каждой половине своего размера (old.size == new.size == half).
            se = float(np.sqrt(old.var() / old.size + new.var() / new.size)) + 1e-8
            if drop / se >= self.lp_z:  # значимо относительно дисперсии канала → не белый шум
                epi[k] = drop
        m = epi.max()
        if m > 0:
            epi = epi / m  # нормировка: один ещё-выучиваемый канал получит всё внимание
        return epi

    def run(self, env: Environment, n_steps: int) -> CuriosityTracker:
        """Прогнать цикл с любопытством на n_steps шагах среды.

        Сбрасывает накопленное состояние в начале — повторный run() на том же объекте
        начинается «с чистого листа», без контаминации предыдущим прогоном.
        """
        self._history = [deque(maxlen=self.context_len + 1) for _ in range(self.K)]
        self._visits = np.zeros(self.K, dtype=int)
        self._err_hist = [deque(maxlen=self.lp_window) for _ in range(self.K)]
        self.tracker = CuriosityTracker(n_actions=self.K)
        env.reset()
        for t in range(n_steps):
            a = self.policy.select_action(self.epistemic_values())

            obs = env.step(a)
            z = self.encoder.encode(np.asarray(obs, dtype=np.float64))
            hist = self._history[a]
            hist.append(z)
            self._visits[a] += 1

            err: float | None = None
            if len(hist) == self.context_len + 1:
                context = np.concatenate([*list(hist)[:-1], self._onehot(a)])  # внутри-канальный контекст + id
                target = hist[-1]

                err = _mse(self.predictor.predict(context), target)
                self._err_hist[a].append(err)
                self.memory.add(Experience(context.copy(), target.copy(), t))

                if t >= self.warmup and t % self.train_every == 0 and len(self.memory) >= self.batch_size:
                    contexts, targets = self.memory.sample(self.batch_size)
                    self.predictor.update(contexts, targets)

            self.tracker.record(t=t, action=a, error=err, epistemic=self.epistemic_values())

        return self.tracker
