"""Думающая система (brain-inspired), улучшающаяся со временем.

MVP реализует ШАГ 1 дорожной карты из research/brain-inspired-thinking-system.md:
замкнутый цикл «предсказание → ошибка → обучение» на одном потоке —

    поток → энкодер → предиктивный движок (предсказывает следующий латент)
          → ошибка предсказания учит предиктор → опыт в эпизодическую память
          → периодический replay из памяти дообучает предиктор.

Каждый архитектурный компонент вынесен в отдельный подпакет с абстрактным
интерфейсом (это «шов» под будущую замену):

    encoders/   — сенсорный ингест (MVP: фиксированная random-projection;
                  шов под обучаемый JEPA-энкодер с EMA-таргетом)
    predictors/ — модель мира / предиктор (MVP: numpy-MLP в латентном пространстве)
    memory/     — эпизодическая память (MVP: кольцевой буфер + replay;
                  шов под иерархическую эпизод/семантик/процедур память)
    streams/    — домен-агностичный источник (MVP: синтетический поток)
    metrics/    — измерение «роста понимания» (ошибка предсказания во времени)
    core/       — Experience + PredictiveLoop (оркестратор цикла)
"""

from thinking_system.core.experience import Experience
from thinking_system.core.loop import PredictiveLoop
from thinking_system.core.curiosity_loop import ActiveInferenceLoop
from thinking_system.encoders.random_projection import RandomProjectionEncoder
from thinking_system.predictors.mlp import MLPPredictor
from thinking_system.predictors.ensemble import EnsemblePredictor
from thinking_system.memory.buffer import EpisodicBuffer
from thinking_system.streams.synthetic import SyntheticStream
from thinking_system.envs.multichannel import MultiChannelEnv
from thinking_system.policies.active_inference import ActiveInferencePolicy, RandomPolicy
from thinking_system.metrics.tracker import MetricsTracker
from thinking_system.metrics.curiosity import CuriosityTracker

__version__ = "0.0.1"

__all__ = [
    # шаг 1 — пассивный предиктивный цикл
    "Experience",
    "PredictiveLoop",
    "RandomProjectionEncoder",
    "MLPPredictor",
    "EpisodicBuffer",
    "SyntheticStream",
    "MetricsTracker",
    # шаг 2 — любопытство / активный вывод
    "ActiveInferenceLoop",
    "EnsemblePredictor",
    "MultiChannelEnv",
    "ActiveInferencePolicy",
    "RandomPolicy",
    "CuriosityTracker",
]
