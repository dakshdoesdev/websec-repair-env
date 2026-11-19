"""Tests for the judge-facing WebSec baseline runner."""

from __future__ import annotations

import websec_repair_env.inference as baseline
from websec_repair_env.orchestration.types import EpisodeResult, StepRecord, TaskSpec


def test_choose_vulnerability_uses_hint_keywords() -> None:
    assert baseline.choose_vulnerability("sqli_login", "SQL injection risk detected") == "sql_injection"
    assert baseline.choose_vulnerability("xss_comments", "Cross-site scripting issue") == "xss"


def test_run_baseline_emits_structured_logs(monkeypatch, capsys) -> None:
    class FakeRunner:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self, *, task: TaskSpec, run_id: str) -> EpisodeResult:
            rewards = [0.05, 0.2, 0.3, 0.35, 0.1]
            steps = [
                StepRecord(
                    index=index,
                    action={"action_type": "inspect" if index == 1 else "submit"},
                    reward=reward,
                    done=index == 5,
                    error=None,
                )
                for index, reward in enumerate(rewards, start=1)
            ]
            return EpisodeResult(
                run_id=run_id,
                episode_id=f"{run_id}:{task.task_id}",
                task_id=task.task_id,
                difficulty=task.difficulty,
                policy_name="local-llm",
                model_name="local-deterministic",
                success=True,
                score=1.0,
                step_count=5,
                failure_type=None,
                failure_reason=None,
                started_at="2026-04-03T00:00:00Z",
                finished_at="2026-04-03T00:00:01Z",
                steps=steps,
            )

    monkeypatch.setattr(baseline, "EpisodeRunner", FakeRunner)

    exit_code = baseline.run_baseline("http://127.0.0.1:8000", "sqli_login", "local-deterministic")
    output = capsys.readouterr().out.strip().splitlines()

    assert exit_code == 0
    assert output[0] == "[START] task=sqli_login env=websec_repair_env model=local-deterministic"
    assert output[-1] == "[END] success=true steps=5 score=1.0 rewards=[0.05, 0.2, 0.3, 0.35, 0.1]"
    assert all(line.startswith("[STEP]") for line in output[1:-1])
