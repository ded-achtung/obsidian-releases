"""Тесты языкового грунтинга: понимание команд и связь с целью."""

from __future__ import annotations

import numpy as np

from thinking_system.language.grounding import BagOfWords, GoalClassifier, generate_commands, goal_cell


def test_goal_cell_mapping() -> None:
    assert goal_cell(0, 7) == (0, 0)        # top-left
    assert goal_cell(3, 7) == (6, 6)        # bottom-right
    assert goal_cell(4, 7) == (3, 3)        # center


def test_classifier_generalizes_to_heldout() -> None:
    data = generate_commands()
    rng = np.random.default_rng(0)
    rng.shuffle(data)
    cut = int(0.7 * len(data))
    train, test = data[:cut], data[cut:]
    clf = GoalClassifier(BagOfWords([t for t, _ in train]))
    clf.fit(train, epochs=300)
    acc = np.mean([clf.predict(t)[0] == c for t, c in test])
    assert acc > 0.9  # обобщает на невиданные формулировки


def test_classifier_understands_fresh_phrasings() -> None:
    data = generate_commands()
    clf = GoalClassifier(BagOfWords([t for t, _ in data]))
    clf.fit(data, epochs=300)
    fresh = [
        ("head over to the upper right", 1),
        ("walk to the lower left", 2),
        ("reach the middle", 4),
        ("proceed to the north west", 0),
        ("move to the south east", 3),
    ]
    for text, cls in fresh:
        assert clf.predict(text)[0] == cls
