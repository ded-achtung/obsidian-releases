"""CognitiveAgent — единый контур думающей системы на реальных байтовых данных.

Сводит вместе все построенные ранее части в ОДИН цикл:

    выбор источника (любопытство, active inference)        ← шаг 2
        ↓
    восприятие байта + предсказание следующего (world model) ← шаг 1
        ↓
    surprise = ошибка предсказания (бит)  →  ИСПОЛЬЗОВАНИЕ:    ← этот шаг
        • высокий surprise = аномалия/новизна;
        • learning progress по surprise = куда направить любопытство;
        ↓
    запись опыта в эпизодическую память + replay-консолидация ← шаг 3
        (большая память = не забывает; маленькая = забывает старое)

Один и тот же сигнал — ошибка предсказания — одновременно (а) учит модель,
(б) направляет внимание и (в) служит детектором аномалий. Это и есть «система
не просто учится, но и использует выученное».
"""

from __future__ import annotations

from collections import deque

import numpy as np

from thinking_system.core.experience import Experience
from thinking_system.memory.buffer import EpisodicBuffer
from thinking_system.policies.active_inference import ActiveInferencePolicy
from thinking_system.predictors.symbolic import SymbolicPredictor
from thinking_system.text.vocab import ByteVocab


class CognitiveAgent:
    """Когнитивный агент, читающий несколько байтовых источников.

    Args:
        sources: список (имя, байтовый массив id) — реальные тексты/код.
        context_len: длина байтового контекста.
        hidden: ширина скрытого слоя модели мира.
        train_every: как часто (в шагах) обучать модель по replay-батчу.
        batch_size: размер replay-батча.
        warmup: шагов до старта обучения.
        gamma: точность политики любопытства.
        memory_capacity: ёмкость эпизодической памяти. Большая = консолидация
            (не забывает); маленькая = только недавнее (забывает старое).
        lp_window/lp_min: окно и минимум данных для оценки learning progress.
        seed: зерно.
    """

    def __init__(
        self,
        sources: list[tuple[str, np.ndarray]],
        *,
        context_len: int = 16,
        hidden: int = 224,
        train_every: int = 2,
        batch_size: int = 64,
        warmup: int = 256,
        gamma: float = 6.0,
        memory_capacity: int = 200_000,
        lp_window: int = 2000,
        lp_min: int = 800,
        seed: int = 0,
    ) -> None:
        self.vocab = ByteVocab()
        self.names = [n for n, _ in sources]
        self.data = [np.asarray(d, dtype=np.int64) for _, d in sources]
        self.K = len(sources)
        self.context_len = context_len
        self.train_every = train_every
        self.batch_size = batch_size
        self.warmup = warmup
        self.lp_window = lp_window
        self.lp_min = lp_min

        self.predictor = SymbolicPredictor(256, context_len, emb_dim=32, hidden_dim=hidden, lr=1e-3, weight_decay=1e-4, seed=seed)
        self.memory = EpisodicBuffer(capacity=memory_capacity, seed=seed)
        self.policy = ActiveInferencePolicy(gamma=gamma, seed=seed)

        self._ptr = [context_len for _ in range(self.K)]          # указатель чтения по источнику
        self._err = [deque(maxlen=lp_window) for _ in range(self.K)]  # окно surprise (для LP)
        self.visits = np.zeros(self.K, dtype=int)
        self.choices: list[int] = []        # какой источник выбран на каждом шаге
        self.bits: list[float] = []         # surprise на каждом шаге
        self._t = 0

    # --- ЛЮБОПЫТСТВО: эпистемическая ценность = значимый learning progress ---
    def epistemic_values(self) -> np.ndarray:
        epi = np.zeros(self.K)
        for k in range(self.K):
            h = self._err[k]
            if len(h) < self.lp_min:
                continue
            a = np.fromiter(h, dtype=np.float64)
            half = a.size // 2
            old, new = a[:half], a[half:]
            drop = float(old.mean() - new.mean())
            se = float(np.sqrt(old.var() / half + new.var() / half)) + 1e-8
            if drop > 0 and drop / se >= 2.0:
                epi[k] = drop
        m = epi.max()
        return epi / m if m > 0 else epi

    def _read(self, k: int) -> tuple[np.ndarray, int]:
        d, L = self.data[k], self.context_len
        p = self._ptr[k]
        if p >= len(d):
            p = L  # дочитали источник — идём по кругу
        ctx = d[p - L : p]
        tgt = int(d[p])
        self._ptr[k] = p + 1
        return ctx, tgt

    def _surprise(self, ctx: np.ndarray, tgt: int) -> float:
        logits = self.predictor.logits(ctx)
        pp = np.exp(logits - logits.max())
        pp /= pp.sum()
        return float(-np.log2(pp[tgt] + 1e-12))

    def step(self) -> tuple[int, float]:
        epi = self.epistemic_values()
        k = self.policy.select_action(epi)           # ВЫБОР источника (любопытство)
        ctx, tgt = self._read(k)                       # ВОСПРИЯТИЕ
        bits = self._surprise(ctx, tgt)                # ПРЕДСКАЗАНИЕ → surprise (ИСПОЛЬЗОВАНИЕ)

        self._err[k].append(bits)
        self.visits[k] += 1
        self.choices.append(k)
        self.bits.append(bits)

        self.memory.add(Experience(ctx.copy(), np.array([tgt]), self._t))  # ПАМЯТЬ
        if self._t >= self.warmup and self._t % self.train_every == 0 and len(self.memory) >= self.batch_size:
            cs, ts = self.memory.sample(self.batch_size)  # КОНСОЛИДАЦИЯ (replay)
            self.predictor.update(cs, ts)

        self._t += 1
        return k, bits

    def run(self, steps: int) -> "CognitiveAgent":
        for _ in range(steps):
            self.step()
        return self

    # --- ИСПОЛЬЗОВАНИЕ: сканировать новый текст и вернуть surprise по байтам (аномалии) ---
    def scan(self, text: str) -> np.ndarray:
        L = self.context_len
        ids = np.concatenate([np.full(L, 32, dtype=np.int64), self.vocab.encode(text)])
        return np.array([self._surprise(ids[i - L : i], int(ids[i])) for i in range(L, len(ids))])

    # --- оценка: текущий BPC по источнику (память/удержание) ---
    def source_bpc(self, k: int, *, n: int = 3000) -> float:
        d, L = self.data[k], self.context_len
        n = min(n, len(d) - L)
        return float(np.mean([self._surprise(d[i - L : i], int(d[i])) for i in range(L, L + n)]))
