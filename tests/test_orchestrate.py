"""Tests for the WebSec orchestrator controller."""

from __future__ import annotations

from pathlib import Path

from websec_repair_env.orchestration import controller
from websec_repair_env.orchestration.types import EpisodeResult, RunConfig, StepRecord, TaskSpec


class FakeRunner:
    def __init__(self, *args, **kwargs) -> None:
        self.calls: list[str] = []

    def run(self, *, task: TaskSpec, run_id: str) -> EpisodeResult:
        self.calls.append(task.task_id)
        success = task.task_id != "broken_auth_admin"
        return EpisodeResult(
            run_id=run_id,
            episode_id=f"{run_id}:{task.task_id}",
            task_id=task.task_id,
            difficulty=task.difficulty,
            policy_name="heuristic",
            model_name="local-test",
            success=success,
            score=1.0 if success else 0.0,
            step_count=3,
            failure_type=None if success else "submission_incomplete",
            failure_reason=None if success else "Submission recorded before a fully verified repair. Episode failed.",
            started_at="2026-04-03T00:00:00Z",
            finished_at="2026-04-03T00:00:01Z",
            steps=[StepRecord(index=1, action={"action_type": "inspect"}, reward=0.05, done=False)],
        )


def _config(tmp_path: Path, mode: str = "smoke") -> RunConfig:
    return RunConfig(
        mode=mode,
        base_url="http://127.0.0.1:8000",
        task_ids=(),
        episodes=2,
        max_steps=6,
        retry_limit=0,
        advance_threshold=0.8,
        policy_name="heuristic",
        model_name="local-test",
        api_base_url="http://127.0.0.1:11434/v1",
        artifacts_dir=str(tmp_path / "runs"),
    )


def test_run_once_smoke_writes_artifacts(monkeypatch, tmp_path: Path) -> None:
    specs = [
        TaskSpec("sqli_login", "easy", "SQLi", "sqli"),
        TaskSpec("xss_comments", "medium", "XSS", "xss"),
    ]
    fake_runner = FakeRunner()
    monkeypatch.setattr(controller, "list_task_specs", lambda task_ids=(): specs)
    monkeypatch.setattr(controller, "EpisodeRunner", lambda **kwargs: fake_runner)

    summary = controller.run_once(_config(tmp_path, mode="smoke"))

    assert summary.total_tasks == 2
    assert summary.total_episodes == 2
    assert (Path(summary.run_dir) / "manifest.json").exists()
    assert (Path(summary.run_dir) / "episodes.jsonl").exists()
    assert (Path(summary.run_dir) / "summary.json").exists()
    assert fake_runner.calls == ["sqli_login", "xss_comments"]


def test_run_once_curriculum_visits_ordered_difficulties(monkeypatch, tmp_path: Path) -> None:
    specs = [
        TaskSpec("sqli_login", "easy", "SQLi", "sqli"),
        TaskSpec("xss_comments", "medium", "XSS", "xss"),
        TaskSpec("broken_auth_admin", "hard", "Broken Auth", "auth"),
    ]
    fake_runner = FakeRunner()
    monkeypatch.setattr(controller, "list_task_specs", lambda task_ids=(): specs)
    monkeypatch.setattr(controller, "EpisodeRunner", lambda **kwargs: fake_runner)

    summary = controller.run_once(_config(tmp_path, mode="curriculum"))

    assert summary.total_tasks == 3
    assert fake_runner.calls[:2] == ["sqli_login", "xss_comments"]
