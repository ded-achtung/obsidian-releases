"""Предиктивный движок: латент-контекст → следующий латент."""

from thinking_system.predictors.base import Predictor
from thinking_system.predictors.mlp import MLPPredictor

__all__ = ["Predictor", "MLPPredictor"]
