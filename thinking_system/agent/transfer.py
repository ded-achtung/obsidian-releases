"""Перенос навыка между мирами: переиспользовать веса, а не учить заново.

Навык — линейная Q(φ(o),a) над признаками восприятия (одинаковыми по размеру миры
→ один энкодер). Поэтому веса, обученные на одном лабиринте, можно ПЕРЕНЕСТИ на
новый:

  • zero-shot — применить веса на невиданном лабиринте без обучения;
  • domain randomization — учить веса на распределении лабиринтов → устойчивый к
    разметке навык, переносящийся на свежие миры;
  • few-shot — тёплый старт: на новом мире навык уже компетентен с первых эпизодов,
    тогда как «с нуля» долго барахтается (jumpstart).
"""

from __future__ import annotations

import numpy as np

from thinking_system.world.gridworld import GridWorld
from thinking_system.agent.latent_qoption import LatentQOption


def _move(grid: GridWorld, s: int, a: int) -> int:
    r, c = divmod(s, grid.size)
    dr, dc = GridWorld.MOVES[a]
    nr, nc = r + dr, c + dc
    if 0 <= nr < grid.size and 0 <= nc < grid.size and (nr, nc) not in grid.walls:
        return grid.sid((nr, nc))
    return s


def transfer_option(grid: GridWorld, subgoal_cell: tuple[int, int], encode, n_features: int, W: np.ndarray, *,
                    alpha: float = 0.05, gamma: float = 0.95, step_penalty: float = 0.01, seed: int = 0) -> LatentQOption:
    """Создать навык на НОВОМ мире (тот же энкодер) с перенесёнными весами W (тёплый старт)."""
    opt = LatentQOption(grid, grid.sid(subgoal_cell), encode, n_features,
                        alpha=alpha, gamma=gamma, step_penalty=step_penalty, seed=seed)
    opt.W = W.copy()
    return opt


def train_across_worlds(grids: list[GridWorld], subgoal_cell: tuple[int, int], encode, n_features: int, *,
                        episodes: int = 8000, eps: float = 0.2, alpha: float = 0.05, gamma: float = 0.95,
                        step_penalty: float = 0.01, max_steps: int = 60, seed: int = 0) -> np.ndarray:
    """Домен-рандомизация: учим ОДНИ веса навыка на распределении миров. Вернуть W."""
    W = np.zeros((4, n_features))
    rng = np.random.default_rng(seed)
    for ep in range(episodes):
        g = grids[ep % len(grids)]
        sub = g.sid(subgoal_cell)
        free = [s for s in range(g.n_states) if (s // g.size, s % g.size) not in g.walls and s != sub]
        s = int(rng.choice(free))
        x = encode(s)
        for _ in range(max_steps):
            a = int(rng.integers(4)) if rng.random() < eps else int(np.argmax(W @ x))
            sp = _move(g, s, a)
            done = sp == sub
            r = 1.0 if done else -step_penalty
            xp = encode(sp)
            target = r + (0.0 if done else gamma * float(np.max(W @ xp)))
            W[a] += alpha * (target - (W @ x)[a]) * x
            s, x = sp, xp
            if done:
                break
    return W


def reach_rate(option: LatentQOption, *, n: int = 40, seed: int = 0) -> float:
    """Доля успешных доходов до подцели из случайных стартов мира навыка."""
    rng = np.random.default_rng(seed)
    free = [s for s in option.free if s != option.subgoal]
    return float(np.mean([option.reach(int(rng.choice(free)), seed=int(rng.integers(10 ** 6))) for _ in range(n)]))
