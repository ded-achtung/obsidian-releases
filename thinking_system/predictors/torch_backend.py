"""Torch-бэкенд предикторов — ТОТ ЖЕ интерфейс, что у numpy-версий, но на GPU.

Эти классы — drop-in замены:
  • TorchMLPPredictor      ↔ MLPPredictor       (интерфейс Predictor: predict/update)
  • TorchSymbolicPredictor ↔ SymbolicPredictor  (logits/nll_bits/update)
Автоопределение устройства (CUDA, если доступна). На GPU тот же код масштабируется
до больших моделей/данных (что нужно для связной генерации) — без изменения
остальной системы.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from thinking_system.predictors.base import Predictor


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TorchMLPPredictor(Predictor):
    """Torch-версия MLPPredictor (предсказание латента). Drop-in для PredictiveLoop."""

    def __init__(self, context_dim: int, latent_dim: int, *, hidden_dim: int = 128, lr: float = 1e-3, seed: int = 0, device=None) -> None:
        torch.manual_seed(seed)
        self.device = device or default_device()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, latent_dim)
        ).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def predict(self, context: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.atleast_2d(context), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.net(x).cpu().numpy()[0]

    def update(self, contexts: np.ndarray, targets: np.ndarray) -> float:
        X = torch.as_tensor(np.atleast_2d(contexts), dtype=torch.float32, device=self.device)
        Y = torch.as_tensor(np.atleast_2d(targets), dtype=torch.float32, device=self.device)
        loss = ((self.net(X) - Y) ** 2).sum(dim=1).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())


class TorchSymbolicPredictor:
    """Torch-версия SymbolicPredictor (следующий символ/байт). GPU-ready для масштаба."""

    def __init__(self, vocab_size: int, context_len: int, *, emb_dim: int = 24, hidden_dim: int = 128, lr: float = 1e-3, weight_decay: float = 0.0, seed: int = 0, device=None) -> None:
        torch.manual_seed(seed)
        self.device = device or default_device()
        self.context_len = context_len
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, emb_dim).to(self.device)
        self.net = nn.Sequential(
            nn.Linear(context_len * emb_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, vocab_size)
        ).to(self.device)
        self.opt = torch.optim.Adam(list(self.emb.parameters()) + list(self.net.parameters()), lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()

    def _logits(self, ctx: torch.Tensor) -> torch.Tensor:
        e = self.emb(ctx).reshape(ctx.shape[0], -1)
        return self.net(e)

    def logits(self, ctx: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.atleast_2d(ctx), dtype=torch.long, device=self.device)
        with torch.no_grad():
            return self._logits(x).cpu().numpy()[0]

    def nll_bits(self, ctx: np.ndarray, targets: np.ndarray) -> float:
        x = torch.as_tensor(np.atleast_2d(ctx), dtype=torch.long, device=self.device)
        y = torch.as_tensor(np.asarray(targets).reshape(-1), dtype=torch.long, device=self.device)
        with torch.no_grad():
            loss = self.loss_fn(self._logits(x), y)
        return float(loss.item()) / np.log(2)

    def update(self, ctx: np.ndarray, targets: np.ndarray) -> float:
        x = torch.as_tensor(np.atleast_2d(ctx), dtype=torch.long, device=self.device)
        y = torch.as_tensor(np.asarray(targets).reshape(-1), dtype=torch.long, device=self.device)
        loss = self.loss_fn(self._logits(x), y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item()) / np.log(2)
