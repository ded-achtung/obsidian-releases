"""Лёгкая визуализация: ASCII-спарклайн всегда, matplotlib — если установлен.

ASCII-спарклайн не требует зависимостей и работает в любом терминале. PNG-график
рисуется опционально (extras: pip install -e '.[viz]').
"""

from __future__ import annotations

import numpy as np

_BARS = "▁▂▃▄▅▆▇█"


def sparkline(values: np.ndarray | list[float], width: int = 70) -> str:
    """ASCII-спарклайн ряда (даунсэмпл до width точек)."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return ""
    if arr.size > width:
        idx = np.linspace(0, arr.size - 1, width).astype(int)
        arr = arr[idx]
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return _BARS[0] * arr.size
    norm = (arr - lo) / (hi - lo)
    return "".join(_BARS[min(len(_BARS) - 1, int(v * (len(_BARS) - 1)))] for v in norm)


def maybe_save_plot(tracker, path: str = "prediction_error.png") -> str | None:
    """Сохранить график ошибки в PNG, если доступен matplotlib. Иначе None."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    roll = tracker.rolling()
    roll_base = tracker.rolling(tracker.baselines)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(roll, label="ошибка предиктора (скольз.)", color="tab:blue")
    ax.plot(roll_base, label="baseline (persistence)", color="tab:gray", linestyle="--")
    ax.set_xlabel("шаг")
    ax.set_ylabel("MSE в латентном пространстве")
    ax.set_title("Рост понимания: ошибка предсказания во времени")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
