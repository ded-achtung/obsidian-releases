"""LanguageOptionAgent — язык ВЫБИРАЕТ выученный навык-опцию.

Команда грунтится в проём (язык → подцель), агент выбирает СООТВЕТСТВУЮЩУЮ
обученную Q-опцию и исполняет её. Цепочка команд → цепочка опций: язык задаёт
последовательность подцелей, агент строит из выученных навыков верхнеуровневый
маршрут (иерархия из языка).

В отличие от MegaAgent (планирование по карте), здесь поведение — это композиция
готовых НАВЫКОВ, выбранных языком: «как назвать, так и сделать».
"""

from __future__ import annotations

from thinking_system.agent.qoption import QOptionLibrary
from thinking_system.language.doorways import doorway_cell


class LanguageOptionAgent:
    """Связывает грунтинг команд с библиотекой Q-обученных опций.

    Args:
        library: QOptionLibrary с опциями к клеткам-проёмам.
        classifier: обученный классификатор команда → индекс проёма.
    """

    def __init__(self, library: QOptionLibrary, classifier) -> None:
        self.lib = library
        self.clf = classifier

    def ground(self, command: str) -> tuple[int, int, float]:
        """Команда → (индекс проёма, клетка-подцель, уверенность)."""
        idx, conf = self.clf.predict(command)
        return idx, self.lib.g.sid(doorway_cell(idx)), conf

    def obey(self, command: str, start: int, *, perturb: float = 0.0, seed: int = 0) -> dict:
        """Понять команду → выбрать опцию → исполнить навык до проёма."""
        idx, subgoal, conf = self.ground(command)
        reached = self.lib.options[subgoal].reach(start, perturb=perturb, seed=seed)
        return {"command": command, "doorway": idx, "subgoal": subgoal, "confidence": conf, "reached": reached}

    def obey_sequence(self, commands: list[str], start: int, *, perturb: float = 0.0, seed: int = 0) -> dict:
        """Цепочка команд → цепочка опций: иерархический маршрут из выученных навыков."""
        seq = [self.ground(c)[1] for c in commands]
        reached = self.lib.compose(start, seq, perturb=perturb, seed=seed)
        return {"commands": commands, "subgoals": seq, "reached": reached}
