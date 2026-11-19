"""Task discovery helpers for orchestrated WebSec runs."""

from __future__ import annotations

from ..server.challenge import TASKS
from .types import TaskSpec

DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}


def list_task_specs(task_ids: tuple[str, ...] = ()) -> list[TaskSpec]:
    available = {
        task_id: TaskSpec(
            task_id=definition["definition"].id,
            difficulty=definition["definition"].difficulty,
            title=definition["definition"].title,
            description=definition["definition"].instruction,
            default_episode_budget=1,
            tags=(definition["correct_vulnerability"],),
        )
        for task_id, definition in TASKS.items()
    }
    if task_ids:
        missing = [task_id for task_id in task_ids if task_id not in available]
        if missing:
            raise ValueError(f"Unknown task ids: {', '.join(missing)}")
        specs = [available[task_id] for task_id in task_ids]
    else:
        specs = list(available.values())
    return sorted(specs, key=lambda item: (DIFFICULTY_ORDER.get(item.difficulty, 99), item.task_id))
