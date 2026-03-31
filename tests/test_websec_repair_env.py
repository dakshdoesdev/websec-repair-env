"""Tests for the deterministic WebSec Repair environment."""

from __future__ import annotations

from fastapi.testclient import TestClient

from websec_repair_env.models import WebSecRepairAction
from websec_repair_env.server.app import app
from websec_repair_env.server.websec_repair_environment import WebSecRepairEnvironment


TASK_CASES = [
    ("sqli_login", "sql_injection", "parameterized_query", "strip_quotes"),
    ("xss_comments", "xss", "html_escape", "remove_script_substring"),
    ("broken_auth_admin", "broken_auth", "require_admin_role", "hide_admin_link"),
]


def _solve_task(env: WebSecRepairEnvironment, task_id: str, vulnerability: str, patch_id: str):
    env.reset(task_id=task_id)
    env.step(WebSecRepairAction(action_type="inspect"))
    env.step(
        WebSecRepairAction(
            action_type="classify",
            vulnerability_type=vulnerability,
        )
    )
    env.step(
        WebSecRepairAction(
            action_type="apply_patch",
            patch_id=patch_id,
        )
    )
    env.step(WebSecRepairAction(action_type="verify"))
    return env.step(WebSecRepairAction(action_type="submit"))


def _run_actions(env: WebSecRepairEnvironment, task_id: str, actions: list[WebSecRepairAction]):
    observation = env.reset(task_id=task_id)
    rewards: list[float] = []
    for action in actions:
        observation = env.step(action)
        rewards.append(float(observation.reward))
        if observation.done:
            break
    return observation, rewards


def test_reset_and_state_smoke() -> None:
    env = WebSecRepairEnvironment()
    obs = env.reset(task_id="xss_comments")
    assert obs.task_id == "xss_comments"
    assert obs.code_snippet == ""
    assert env.state.inspected is False
    assert env.state.workflow_stage == "start"

    env.step(WebSecRepairAction(action_type="inspect"))
    assert env.state.inspected is True
    assert env.state.task_id == "xss_comments"
    assert env.state.workflow_stage == "inspected"

    obs = env.reset(task_id="broken_auth_admin")
    assert obs.task_id == "broken_auth_admin"
    assert env.state.inspected is False
    assert env.state.selected_vulnerability == ""
    assert env.state.applied_patch_id == ""
    assert env.state.exploit_test_passed is False


def test_happy_path_for_each_task() -> None:
    for task_id, vulnerability, correct_patch, _ in TASK_CASES:
        env = WebSecRepairEnvironment()
        result = _solve_task(env, task_id, vulnerability, correct_patch)
        assert result.done is True
        assert result.grader_passed is True
        assert result.exploit_test_passed is True
        assert result.functionality_test_passed is True
        assert env.state.score == 1.0


def test_wrong_patch_failure_for_each_task() -> None:
    for task_id, vulnerability, _, wrong_patch in TASK_CASES:
        env = WebSecRepairEnvironment()
        result = _solve_task(env, task_id, vulnerability, wrong_patch)
        assert result.done is True
        assert result.grader_passed is False
        assert env.state.score < 1.0
        assert (
            result.exploit_test_passed is False
            or result.functionality_test_passed is False
        )


def test_inspect_reveals_context_and_repeat_is_penalized() -> None:
    env = WebSecRepairEnvironment()
    reset_obs = env.reset(task_id="sqli_login")
    assert reset_obs.code_snippet == ""
    assert reset_obs.scanner_hint == ""
    assert reset_obs.patch_options == []

    first_inspect = env.step(WebSecRepairAction(action_type="inspect"))
    assert first_inspect.reward > 0
    assert first_inspect.code_snippet
    assert first_inspect.scanner_hint
    assert first_inspect.patch_options
    assert first_inspect.workflow_stage == "inspected"

    second_inspect = env.step(WebSecRepairAction(action_type="inspect"))
    assert second_inspect.reward < 0
    assert "repeated" in second_inspect.status_message.lower()


def test_perfect_trajectory_beats_shortcut_and_random_play() -> None:
    perfect_env = WebSecRepairEnvironment()
    perfect_result, perfect_rewards = _run_actions(
        perfect_env,
        "sqli_login",
        [
            WebSecRepairAction(action_type="inspect"),
            WebSecRepairAction(action_type="classify", vulnerability_type="sql_injection"),
            WebSecRepairAction(action_type="apply_patch", patch_id="parameterized_query"),
            WebSecRepairAction(action_type="verify"),
            WebSecRepairAction(action_type="submit"),
        ],
    )

    shortcut_env = WebSecRepairEnvironment()
    shortcut_result, shortcut_rewards = _run_actions(
        shortcut_env,
        "sqli_login",
        [
            WebSecRepairAction(action_type="apply_patch", patch_id="parameterized_query"),
            WebSecRepairAction(action_type="verify"),
            WebSecRepairAction(action_type="submit"),
        ],
    )

    random_env = WebSecRepairEnvironment()
    random_result, random_rewards = _run_actions(
        random_env,
        "sqli_login",
        [
            WebSecRepairAction(action_type="inspect"),
            WebSecRepairAction(action_type="inspect"),
            WebSecRepairAction(action_type="submit"),
        ],
    )

    assert perfect_result.grader_passed is True
    assert perfect_env.state.score == 1.0
    assert sum(perfect_rewards) > sum(random_rewards)
    assert sum(perfect_rewards) > sum(shortcut_rewards)
    assert sum(shortcut_rewards) < 0
    assert sum(random_rewards) != sum(shortcut_rewards)
    assert shortcut_result.done is True
    assert random_result.done is True


def test_http_routes_return_expected_shapes() -> None:
    client = TestClient(app)

    tasks_response = client.get("/tasks")
    assert tasks_response.status_code == 200
    tasks_payload = tasks_response.json()
    assert tasks_payload["environment"] == "websec_repair_env"
    assert len(tasks_payload["tasks"]) == 3
    assert tasks_payload["available_difficulties"] == ["easy", "hard", "medium"]

    baseline_response = client.get("/baseline", params={"task_id": "sqli_login"})
    assert baseline_response.status_code == 200
    baseline_payload = baseline_response.json()
    assert len(baseline_payload["baselines"]) == 1
    assert baseline_payload["baselines"][0]["task_id"] == "sqli_login"

    grader_response = client.get("/grader", params={"task_id": "sqli_login"})
    assert grader_response.status_code == 200
    grader_payload = grader_response.json()
    assert grader_payload["task_id"] == "sqli_login"
    assert "checks" in grader_payload

    filtered_tasks_response = client.get("/tasks", params={"difficulty": "easy"})
    assert filtered_tasks_response.status_code == 200
    filtered_tasks_payload = filtered_tasks_response.json()
    assert filtered_tasks_payload["filtered_difficulty"] == "easy"
    assert len(filtered_tasks_payload["tasks"]) == 1
    assert filtered_tasks_payload["tasks"][0]["id"] == "sqli_login"

    invalid_difficulty_response = client.get("/tasks", params={"difficulty": "expert"})
    assert invalid_difficulty_response.status_code == 404

    status_response = client.get("/status")
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["environment"] == "websec_repair_env"
    assert "progress" in status_payload
    assert "grader" in status_payload
