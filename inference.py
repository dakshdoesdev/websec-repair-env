#!/usr/bin/env python3
"""Deterministic baseline runner for the WebSec Repair environment."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    PARENT = Path(__file__).resolve().parent.parent
    if str(PARENT) not in sys.path:
        sys.path.insert(0, str(PARENT))
    from websec_repair_env.baseline_policy import (
        DEFAULT_MODEL_NAME,
        LocalLLMPolicy,
        TASK_TO_VULNERABILITY,
        choose_vulnerability,
        build_policy_config,
        format_action_payload,
    )
    from websec_repair_env.client import WebSecRepairEnv
    from websec_repair_env.live_dashboard import BaselineDashboard
    from websec_repair_env.models import WebSecRepairAction
    from websec_repair_env.orchestration.runner import EpisodeRunner
    from websec_repair_env.orchestration.tasks import list_task_specs
    from websec_repair_env.terminal_ui import BaselineTui
else:
    from .baseline_policy import (
        DEFAULT_MODEL_NAME,
        LocalLLMPolicy,
        TASK_TO_VULNERABILITY,
        choose_vulnerability,
        build_policy_config,
        format_action_payload,
    )
    from .client import WebSecRepairEnv
    from .live_dashboard import BaselineDashboard
    from .models import WebSecRepairAction
    from .orchestration.runner import EpisodeRunner
    from .orchestration.tasks import list_task_specs
    from .terminal_ui import BaselineTui


def print_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}")


def print_step(step: int, action: WebSecRepairAction, reward: float, done: bool, error: str | None) -> None:
    print(
        "[STEP] "
        f"step={step} "
        f"action={format_action_payload(action)} "
        f"reward={reward} "
        f"done={str(done).lower()} "
        f"error={error or 'null'}"
    )


def print_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    print(
        "[END] "
        f"success={str(success).lower()} "
        f"steps={steps} "
        f"score={score} "
        f"rewards={json.dumps(rewards)}"
    )


def _task_title(task_id: str) -> str:
    return {
        "sqli_login": "Easy - SQLi Login",
        "xss_comments": "Medium - XSS Comments",
        "broken_auth_admin": "Hard - Broken Auth",
    }.get(task_id, task_id)


def _task_difficulty(task_id: str) -> str:
    return {
        "sqli_login": "easy",
        "xss_comments": "medium",
        "broken_auth_admin": "hard",
    }.get(task_id, "unknown")


class _NullDashboard:
    def __enter__(self) -> "_NullDashboard":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def _connect_env(base_url: str) -> WebSecRepairEnv:
    attempts = 10
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            env = WebSecRepairEnv(base_url=base_url).sync()
            env.__enter__()
            return env
        except Exception as exc:  # pragma: no cover - runtime guard
            last_error = exc
            time.sleep(0.35)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to connect to environment")


def run_baseline(
    base_url: str,
    task_id: str,
    model_name: str,
    *,
    plain: bool = True,
    rich: bool = False,
) -> int:
    if not plain and not rich and sys.stdout.isatty():
        return BaselineTui(base_url=base_url, task_id=task_id, model_name=model_name).run()

    use_dashboard = rich and not plain and sys.stdout.isatty()
    dashboard = BaselineDashboard(model_name=model_name, task_id=task_id) if use_dashboard else None
    dashboard_cm = dashboard if dashboard is not None else _NullDashboard()
    task_lookup = {task.task_id: task for task in list_task_specs()}
    runner = EpisodeRunner(
        base_url=base_url,
        policy_name="local-llm",
        model_name=model_name,
    )

    with dashboard_cm:
        result = runner.run(task=task_lookup[task_id], run_id="baseline")
        rewards = [step.reward for step in result.steps]
        if dashboard is not None:
            dashboard.start_episode(
                title=_task_title(task_id),
                difficulty=_task_difficulty(task_id),
                max_steps=runner.max_steps,
            )
            total_reward = 0.0
            for step in result.steps:
                total_reward += step.reward
                observation = {
                    "task_id": task_id,
                    "status_message": step.status_message,
                    "metadata": {"max_steps": runner.max_steps},
                }
                dashboard.record_step(
                    step_count=step.index,
                    action=f"{step.metadata.get('source', 'runner')} {format_action_payload(WebSecRepairAction(**step.action))}",
                    observation=observation,
                    reward=step.reward,
                    total_reward=total_reward,
                )
            dashboard.finish_episode(success=result.success, score=result.score)
            print("\n=== BASELINE SUMMARY ===")
            print(f"task={task_id} model={model_name} base_url={base_url}")
            print(f"success={result.success} steps={result.step_count} score={result.score:.2f}")
        else:
            print_start(task=task_id, env="websec_repair_env", model=model_name)
            for step in result.steps:
                print_step(step.index, WebSecRepairAction(**step.action), step.reward, step.done, step.error)
            print_end(success=result.success, steps=result.step_count, score=result.score, rewards=rewards)
        return 0 if result.success else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--task",
        default="sqli_login",
        choices=sorted(TASK_TO_VULNERABILITY),
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Label used in structured logs for the baseline runner.",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Disable the Rich terminal dashboard and print only structured logs.",
    )
    parser.add_argument(
        "--rich",
        action="store_true",
        help="Use the Rich auto-run dashboard instead of the persistent curses TUI.",
    )
    args = parser.parse_args()
    raise SystemExit(run_baseline(args.base_url, args.task, args.model_name, plain=args.plain, rich=args.rich))


if __name__ == "__main__":
    main()
