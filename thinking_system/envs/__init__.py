"""Управляемые среды: агент выбирает действие → получает наблюдение.

В отличие от streams/ (пассивный поток), здесь агент АКТИВЕН — он решает, что
наблюдать дальше. Это предпосылка для активного вывода / любопытства (шаг 2).
"""

from thinking_system.envs.base import Environment
from thinking_system.envs.multichannel import MultiChannelEnv

__all__ = ["Environment", "MultiChannelEnv"]
