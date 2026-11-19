"""Structured types for orchestrated WebSec runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from statistics import mean
from typing import Any


def utc_now() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    difficulty: str
    title: str
    description: str
    default_episode_budget: int = 1
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "title": self.title,
            "description": self.description,
            "default_episode_budget": self.default_episode_budget,
            "tags": list(self.tags),
        }


@dataclass
class StepRecord:
    index: int
    action: dict[str, Any]
    reward: float
    done: bool
    error: str | None = None
    status_message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "action": self.action,
            "reward": self.reward,
            "done": self.done,
            "error": self.error,
            "status_message": self.status_message,
            "metadata": self.metadata,
        }


@dataclass
class EpisodeResult:
    run_id: str
    episode_id: str
    task_id: str
    difficulty: str
    policy_name: str
    model_name: str
    success: bool
    score: float
    step_count: int
    failure_type: str | None
    failure_reason: str | None
    started_at: str
    finished_at: str
    steps: list[StepRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "policy_name": self.policy_name,
            "model_name": self.model_name,
            "success": self.success,
            "score": self.score,
            "step_count": self.step_count,
            "failure_type": self.failure_type,
            "failure_reason": self.failure_reason,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass
class TaskAggregate:
    task_id: str
    difficulty: str
    episodes_run: int
    success_count: int
    success_rate: float
    average_score: float
    average_steps: float
    failure_counts: dict[str, int]

    @classmethod
    def from_results(cls, task: TaskSpec, results: list[EpisodeResult]) -> "TaskAggregate":
        failure_counts: dict[str, int] = {}
        for result in results:
            key = result.failure_type or "success"
            failure_counts[key] = failure_counts.get(key, 0) + 1
        return cls(
            task_id=task.task_id,
            difficulty=task.difficulty,
            episodes_run=len(results),
            success_count=sum(1 for result in results if result.success),
            success_rate=round(sum(1 for result in results if result.success) / len(results), 4) if results else 0.0,
            average_score=round(mean(result.score for result in results), 4) if results else 0.0,
            average_steps=round(mean(result.step_count for result in results), 2) if results else 0.0,
            failure_counts=failure_counts,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "episodes_run": self.episodes_run,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "average_score": self.average_score,
            "average_steps": self.average_steps,
            "failure_counts": self.failure_counts,
        }


@dataclass
class RunSummary:
    run_id: str
    mode: str
    policy_name: str
    model_name: str
    started_at: str
    finished_at: str
    total_tasks: int
    total_episodes: int
    success_rate: float
    task_aggregates: list[TaskAggregate]
    run_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "policy_name": self.policy_name,
            "model_name": self.model_name,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_tasks": self.total_tasks,
            "total_episodes": self.total_episodes,
            "success_rate": self.success_rate,
            "run_dir": self.run_dir,
            "task_aggregates": [item.to_dict() for item in self.task_aggregates],
        }


@dataclass(frozen=True)
class RunConfig:
    mode: str
    base_url: str
    task_ids: tuple[str, ...]
    episodes: int
    max_steps: int
    retry_limit: int
    advance_threshold: float
    policy_name: str
    model_name: str
    api_base_url: str
    artifacts_dir: str
    watch: bool = False
    watch_interval: int = 300
    models: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "base_url": self.base_url,
            "task_ids": list(self.task_ids),
            "episodes": self.episodes,
            "max_steps": self.max_steps,
            "retry_limit": self.retry_limit,
            "advance_threshold": self.advance_threshold,
            "policy_name": self.policy_name,
            "model_name": self.model_name,
            "api_base_url": self.api_base_url,
            "artifacts_dir": self.artifacts_dir,
            "watch": self.watch,
            "watch_interval": self.watch_interval,
            "models": list(self.models),
        }
