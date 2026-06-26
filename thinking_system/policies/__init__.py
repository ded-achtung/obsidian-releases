"""Выбор действий: какое наблюдение запросить следующим."""

from thinking_system.policies.active_inference import ActiveInferencePolicy, RandomPolicy
from thinking_system.policies.base import Policy

__all__ = ["Policy", "ActiveInferencePolicy", "RandomPolicy"]
