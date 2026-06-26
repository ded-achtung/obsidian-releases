"""Torch char-LSTM — рекуррентная модель для СВЯЗНОЙ генерации текста/кода.

В отличие от предиктора с фиксированным окном (даёт «словесную кашу»), LSTM держит
контекст в скрытом состоянии и генерирует связные слова/строки. Тот же домен-
агностичный байтовый/символьный вход. Устройство авто (CUDA при наличии); на GPU
тот же код масштабируется до больших моделей и длинных контекстов.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from thinking_system.predictors.torch_backend import default_device


class CharRNN:
    """Символьный LSTM-генератор (embedding → LSTM → softmax над словарём)."""

    def __init__(self, vocab_size: int, *, emb: int = 64, hidden: int = 256, layers: int = 2, lr: float = 2e-3, seed: int = 0, device=None) -> None:
        torch.manual_seed(seed)
        self.device = device or default_device()
        self.vocab = vocab_size
        self.emb = nn.Embedding(vocab_size, emb).to(self.device)
        self.lstm = nn.LSTM(emb, hidden, layers, batch_first=True).to(self.device)
        self.fc = nn.Linear(hidden, vocab_size).to(self.device)
        self.opt = torch.optim.Adam(self._params(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def _params(self):
        return list(self.emb.parameters()) + list(self.lstm.parameters()) + list(self.fc.parameters())

    def _forward(self, x, h=None):
        out, h = self.lstm(self.emb(x), h)
        return self.fc(out), h

    def fit(self, ids: np.ndarray, *, seq_len: int = 48, batch: int = 32, steps: int = 6000, log_every: int = 0):
        data = torch.as_tensor(np.asarray(ids), dtype=torch.long, device=self.device)
        n = len(data)
        for step in range(steps):
            ix = torch.randint(0, n - seq_len - 1, (batch,), device=self.device)
            x = torch.stack([data[i : i + seq_len] for i in ix])
            y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])
            logits, _ = self._forward(x)
            loss = self.loss_fn(logits.reshape(-1, self.vocab), y.reshape(-1))
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._params(), 5.0)
            self.opt.step()
            if log_every and (step + 1) % log_every == 0:
                print(f"   step {step + 1}: train bpc {float(loss.item()) / np.log(2):.2f}")
        return self

    @torch.no_grad()
    def bpc(self, ids: np.ndarray, *, seq_len: int = 128) -> float:
        data = torch.as_tensor(np.asarray(ids), dtype=torch.long, device=self.device)
        total = count = 0
        for i in range(0, len(data) - seq_len - 1, seq_len):
            x = data[i : i + seq_len][None]
            y = data[i + 1 : i + seq_len + 1][None]
            logits, _ = self._forward(x)
            total += float(self.loss_fn(logits.reshape(-1, self.vocab), y.reshape(-1)).item()) * seq_len
            count += seq_len
        return total / max(count, 1) / np.log(2)

    @torch.no_grad()
    def generate(self, seed_ids, n: int, *, temp: float = 0.6) -> list[int]:
        self.lstm.flatten_parameters()
        x = torch.as_tensor(np.asarray(seed_ids), dtype=torch.long, device=self.device)[None]
        logits, h = self._forward(x)
        last = logits[0, -1]
        out: list[int] = []
        for _ in range(n):
            p = torch.softmax(last / temp, dim=-1)
            nxt = int(torch.multinomial(p, 1))
            out.append(nxt)
            logits, h = self._forward(torch.tensor([[nxt]], device=self.device), h)
            last = logits[0, -1]
        return out
