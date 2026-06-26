"""Связь действия с языком: текстовая команда → цель, которую агент достигает."""

from thinking_system.language.grounding import (
    BagOfWords,
    GoalClassifier,
    generate_commands,
    goal_cell,
)

__all__ = ["BagOfWords", "GoalClassifier", "generate_commands", "goal_cell"]
