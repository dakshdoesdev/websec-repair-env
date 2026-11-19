"""Artifact and summary helpers for orchestrated WebSec runs."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .types import EpisodeResult, RunConfig, RunSummary


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "run"


def create_run_dir(root: str, run_id: str, mode: str, label: str) -> Path:
    run_dir = Path(root) / f"{run_id}-{_slug(mode)}-{_slug(label)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_manifest(run_dir: Path, config: RunConfig) -> None:
    (run_dir / "manifest.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def append_episode(run_dir: Path, result: EpisodeResult) -> None:
    with (run_dir / "episodes.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result.to_dict(), sort_keys=True))
        handle.write("\n")


def write_summary(run_dir: Path, summary: RunSummary) -> None:
    (run_dir / "summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "summary.md").write_text(render_markdown_summary(summary), encoding="utf-8")


def render_terminal_summary(summary: RunSummary) -> str:
    lines = [
        "",
        "=== ORCHESTRATE SUMMARY ===",
        f"mode={summary.mode} policy={summary.policy_name} model={summary.model_name}",
        f"episodes={summary.total_episodes} tasks={summary.total_tasks} success_rate={summary.success_rate:.2%}",
    ]
    for aggregate in summary.task_aggregates:
        lines.append(
            f"{aggregate.task_id}: success_rate={aggregate.success_rate:.2%} "
            f"avg_score={aggregate.average_score:+.2f} avg_steps={aggregate.average_steps:.2f}"
        )
    lines.append(f"artifacts={summary.run_dir}")
    return "\n".join(lines)


def render_markdown_summary(summary: RunSummary) -> str:
    lines = [
        "# Orchestrate Summary",
        "",
        f"- Run ID: `{summary.run_id}`",
        f"- Mode: `{summary.mode}`",
        f"- Policy: `{summary.policy_name}`",
        f"- Model: `{summary.model_name}`",
        f"- Success rate: `{summary.success_rate:.2%}`",
        "",
        "| Task | Difficulty | Episodes | Success Rate | Avg Score | Avg Steps |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for aggregate in summary.task_aggregates:
        lines.append(
            f"| `{aggregate.task_id}` | `{aggregate.difficulty}` | {aggregate.episodes_run} | "
            f"{aggregate.success_rate:.2%} | {aggregate.average_score:+.2f} | {aggregate.average_steps:.2f} |"
        )
    return "\n".join(lines) + "\n"
