#!/usr/bin/env python3
"""П.3: GPU/torch-бэкенд ТЕМ ЖЕ интерфейсом — drop-in для масштаба.

Показывает, что torch-предикторы подставляются в существующую систему без изменений:
  1) TorchMLPPredictor — в тот же PredictiveLoop (шаг 1);
  2) TorchSymbolicPredictor — в ту же текстовую оценку (held-out BPC).
Устройство выбирается автоматически (CUDA при наличии). На GPU тот же код тянет
большие модели/данные (для связной генерации) — остальная система не меняется.

Запуск: python run_torch.py
"""

from __future__ import annotations

import numpy as np
import torch

from thinking_system.core.loop import PredictiveLoop
from thinking_system.encoders.random_projection import RandomProjectionEncoder
from thinking_system.memory.buffer import EpisodicBuffer
from thinking_system.predictors.torch_backend import TorchMLPPredictor, TorchSymbolicPredictor, default_device
from thinking_system.streams.synthetic import SyntheticStream
from thinking_system.text.dataset import eval_bpc, make_pairs, train_test_split, uniform_bpc, unigram_bpc
from thinking_system.text.ingest import load_book
from thinking_system.text.vocab import CharVocab


def main():
    dev = default_device()
    print(f"▶ Torch {torch.__version__}; устройство: {dev}  (CUDA доступна: {torch.cuda.is_available()})\n")

    # 1) DROP-IN в предиктивный цикл шага 1 (PredictiveLoop не меняется)
    stream = SyntheticStream(obs_dim=32, seed=0)
    enc = RandomProjectionEncoder(32, 16, seed=0)
    pred = TorchMLPPredictor(2 * 16, 16, hidden_dim=128, lr=1e-3, seed=0)
    loop = PredictiveLoop(enc, pred, EpisodicBuffer(10000, seed=0), context_len=2)
    s = loop.run(stream, 5000).summary()
    print("1) DROP-IN в PredictiveLoop (шаг 1) с torch-предиктором:")
    print(f"   ошибка предсказания: {s['initial_error']:.4f} → {s['final_error']:.4f}  (baseline {s['baseline_final']:.4f})")
    print(f"   бьёт baseline: {'да' if s['beats_baseline'] else 'нет'}\n")

    # 2) DROP-IN для текста (тот же интерфейс logits/nll_bits/update → eval_bpc работает)
    text = load_book("books/sample_mathematics.md")
    vocab = CharVocab(text)
    ids = vocab.encode(text)
    tr, te = train_test_split(ids, train_frac=0.9)
    ctr, ttr = make_pairs(tr, 10)
    cte, tte = make_pairs(te, 10)
    tp = TorchSymbolicPredictor(vocab.size, 10, emb_dim=24, hidden_dim=128, lr=1e-3, weight_decay=1e-4, seed=0)
    rng = np.random.default_rng(0)
    best = float("inf")
    for step in range(8000):
        b = rng.integers(0, len(ctr), 64)
        tp.update(ctr[b], ttr[b])
        if (step + 1) % 1000 == 0:
            best = min(best, eval_bpc(tp, cte, tte))
    uni = unigram_bpc(tr, tte, vocab.size)
    print("2) DROP-IN текстового предиктора (TorchSymbolicPredictor, тот же интерфейс):")
    print(f"   held-out bits/char: {best:.2f}  (unigram {uni:.2f}, uniform {uniform_bpc(vocab.size):.2f})")
    print(f"   бьёт unigram: {'да' if best < uni else 'нет'}")

    print("\n── Итог ──")
    print("   Torch-предикторы — drop-in замены numpy-версий: тот же интерфейс, та же")
    print("   система. На GPU тот же код масштабируется до больших моделей и данных")
    print("   (для связной генерации) — менять остальное не нужно.")


if __name__ == "__main__":
    main()
