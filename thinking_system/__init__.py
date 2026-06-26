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
from thinking_system.metrics.continual import ContinualEvaluator
from thinking_system.tasks.sequence import Task, make_task_sequence
from thinking_system.consolidation.continual import ContinualLearner
from thinking_system.predictors.symbolic import SymbolicPredictor
from thinking_system.text.ingest import load_book, load_corpus
from thinking_system.text.vocab import ByteVocab, CharVocab
from thinking_system.agent.cognitive import CognitiveAgent
from thinking_system.agent.acting import ActingAgent
from thinking_system.agent.belief import BeliefAgent
from thinking_system.agent.unified import UnifiedAgent
from thinking_system.agent.mega import MegaAgent
from thinking_system.agent.intrinsic import IntrinsicAgent
from thinking_system.agent.autotelic import AutotelicAgent
from thinking_system.agent.qoption import QOption, QOptionLibrary
from thinking_system.memory.skill_policies import OptionPolicies
from thinking_system.world.gridworld import GridWorld, default_maze
from thinking_system.world.latent_model import LatentWorldModel
from thinking_system.world.partial import PartialGridWorld
from thinking_system.world.noisy_partial import NoisyPartialWorld, unified_maze
from thinking_system.world.rooms import rooms_world
from thinking_system.world.features import CellFeatures, FeatureWorld
from thinking_system.memory.hierarchical import HierarchicalMemory
from thinking_system.language.grounding import GoalClassifier, generate_commands, goal_cell

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
    # шаг 3 — консолидация / непрерывное обучение
    "ContinualLearner",
    "ContinualEvaluator",
    "Task",
    "make_task_sequence",
    # реальные данные — обучение по книгам (текст)
    "SymbolicPredictor",
    "load_book",
    "load_corpus",
    "CharVocab",
    "ByteVocab",
    # единый когнитивный контур (восприятие→предсказание→использование→память)
    "CognitiveAgent",
    # действие во внешнем мире (модель мира + активный вывод)
    "ActingAgent",
    "GridWorld",
    "default_maze",
    "LatentWorldModel",
    # частичная наблюдаемость (POMDP): вера + активный вывод
    "PartialGridWorld",
    "BeliefAgent",
    # язык → цель → поведение
    "GoalClassifier",
    "generate_commands",
    "goal_cell",
    # объединённый агент: восприятие + вера + язык в одном контуре
    "UnifiedAgent",
    "NoisyPartialWorld",
    "unified_maze",
    # иерархическая память: эпизод → семантика (ориентиры) → навыки
    "HierarchicalMemory",
    "rooms_world",
    # мега-слияние: JEPA-восприятие + иерархическая память + язык
    "MegaAgent",
    "CellFeatures",
    "FeatureWorld",
    # внутренняя мотивация: агент сам ставит себе цели
    "IntrinsicAgent",
    "AutotelicAgent",   # любопытство движет поведением в шумном мире (JEPA-восприятие)
    "OptionPolicies",   # навыки как политики-опции (иерархический RL)
    # навыки из опыта: Q-learning опции (иерархический RL без готовой карты)
    "QOption",
    "QOptionLibrary",
]
