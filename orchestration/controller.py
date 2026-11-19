"""Run controller for orchestrated WebSec execution."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

from ..baseline_policy import DEFAULT_API_BASE_URL, DEFAULT_MODEL_NAME
from .reporting import append_episode, create_run_dir, render_terminal_summary, write_manifest, write_summary
from .runner import EpisodeRunner
from .tasks import list_task_specs
from .types import EpisodeResult, RunConfig, RunSummary, TaskAggregate, TaskSpec, utc_now


def default_run_config() -> RunConfig:
    return RunConfig(
        mode="smoke",
        base_url="http://127.0.0.1:8000",
        task_ids=(),
        episodes=3,
        max_steps=6,
        retry_limit=0,
        advance_threshold=0.8,
        policy_name="local-llm",
        model_name=DEFAULT_MODEL_NAME,
        api_base_url=DEFAULT_API_BASE_URL,
        artifacts_dir="runs",
    )


def _fixed_results(config: RunConfig, runner: EpisodeRunner, tasks: list[TaskSpec], run_dir: Path, run_id: str) -> list[EpisodeResult]:
    results: list[EpisodeResult] = []
    per_task_episodes = 1 if config.mode == "smoke" else config.episodes
    for task in tasks:
        for _ in range(per_task_episodes):
            result = runner.run(task=task, run_id=run_id)
            append_episode(run_dir, result)
            results.append(result)
    return results


def _curriculum_results(config: RunConfig, runner: EpisodeRunner, tasks: list[TaskSpec], run_dir: Path, run_id: str) -> list[EpisodeResult]:
    results: list[EpisodeResult] = []
    current_difficulty = None
    difficulty_results: list[EpisodeResult] = []
    for task in tasks:
        if current_difficulty != task.difficulty:
            current_difficulty = task.difficulty
            difficulty_results = []
        attempts = 0
        while attempts < config.episodes:
            result = runner.run(task=task, run_id=run_id)
            append_episode(run_dir, result)
            results.append(result)
            difficulty_results.append(result)
            attempts += 1
            success_rate = sum(1 for item in difficulty_results if item.success) / len(difficulty_results)
            if success_rate >= config.advance_threshold:
                break
    return results


def _summarize(config: RunConfig, run_dir: Path, run_id: str, started_at: str, tasks: list[TaskSpec], results: list[EpisodeResult]) -> RunSummary:
    task_aggregates = []
    for task in tasks:
        task_results = [result for result in results if result.task_id == task.task_id]
        task_aggregates.append(TaskAggregate.from_results(task, task_results))
    success_rate = round(sum(1 for result in results if result.success) / len(results), 4) if results else 0.0
    return RunSummary(
        run_id=run_id,
        mode=config.mode,
        policy_name=config.policy_name,
        model_name=config.model_name,
        started_at=started_at,
        finished_at=utc_now(),
        total_tasks=len(tasks),
        total_episodes=len(results),
        success_rate=success_rate,
        task_aggregates=task_aggregates,
        run_dir=str(run_dir),
    )


def run_once(config: RunConfig) -> RunSummary:
    run_id = utc_now().replace(":", "").replace("-", "")
    tasks = list_task_specs(config.task_ids)
    run_dir = create_run_dir(config.artifacts_dir, run_id, config.mode, config.model_name)
    write_manifest(run_dir, config)
    runner = EpisodeRunner(
        base_url=config.base_url,
        policy_name=config.policy_name,
        model_name=config.model_name,
        api_base_url=config.api_base_url,
        max_steps=config.max_steps,
    )
    started_at = utc_now()
    if config.mode == "curriculum":
        results = _curriculum_results(config, runner, tasks, run_dir, run_id)
    else:
        results = _fixed_results(config, runner, tasks, run_dir, run_id)
    summary = _summarize(config, run_dir, run_id, started_at, tasks, results)
    write_summary(run_dir, summary)
    return summary


def run_tournament(config: RunConfig) -> list[RunSummary]:
    models = config.models or (config.model_name,)
    summaries = []
    for model_name in models:
        summary = run_once(replace(config, mode="benchmark", model_name=model_name, watch=False))
        summaries.append(summary)
    return summaries


def run_watch_loop(config: RunConfig) -> None:
    while True:
        summary = run_once(replace(config, watch=False))
        print(render_terminal_summary(summary))
        time.sleep(config.watch_interval)
