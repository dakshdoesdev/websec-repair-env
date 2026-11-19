"""Reusable episode runner for the WebSec baseline and orchestrator."""

from __future__ import annotations

from types import SimpleNamespace

from ..baseline_policy import (
    DEFAULT_API_BASE_URL,
    DEFAULT_MODEL_NAME,
    LocalLLMPolicy,
    apply_guardrails,
    build_policy_config,
    fallback_action,
    format_action_payload,
)
from ..client import WebSecRepairEnv
from .types import EpisodeResult, StepRecord, TaskSpec, utc_now


def _connect_env(base_url: str) -> WebSecRepairEnv:
    env = WebSecRepairEnv(base_url=base_url).sync()
    env.__enter__()
    return env


class EpisodeRunner:
    def __init__(
        self,
        *,
        base_url: str,
        policy_name: str,
        model_name: str = DEFAULT_MODEL_NAME,
        api_base_url: str = DEFAULT_API_BASE_URL,
        max_steps: int = 6,
    ) -> None:
        self.base_url = base_url
        self.policy_name = policy_name
        self.model_name = model_name
        self.api_base_url = api_base_url
        self.max_steps = max_steps
        self.policy = None
        if policy_name != "heuristic":
            self.policy = LocalLLMPolicy(
                build_policy_config(api_base_url=api_base_url, model_name=model_name)
            )

    def _choose_action(self, observation, history):
        if self.policy is None:
            action = fallback_action(observation, history)
            guarded, guardrail = apply_guardrails(observation, action)
            source = "heuristic+guardrail" if guardrail else "heuristic"
            return guarded, source, guardrail
        return self.policy.choose_action(observation, history)

    def run(self, *, task: TaskSpec, run_id: str) -> EpisodeResult:
        started_at = utc_now()
        rewards: list[float] = []
        history: list[dict[str, object]] = []
        steps: list[StepRecord] = []

        env = _connect_env(self.base_url)
        try:
            result = env.reset(task_id=task.task_id)
            while not result.done and len(steps) < self.max_steps:
                action, source, error = self._choose_action(result.observation, history)
                result = env.step(action)
                reward = float(result.reward or 0.0)
                rewards.append(reward)
                history.append(
                    {
                        "step": len(steps) + 1,
                        "action": format_action_payload(action),
                        "reward": reward,
                        "status_message": result.observation.status_message,
                        "source": source,
                    }
                )
                steps.append(
                    StepRecord(
                        index=len(steps) + 1,
                        action=action.model_dump(exclude_none=True),
                        reward=reward,
                        done=result.done,
                        error=error,
                        status_message=result.observation.status_message,
                        metadata={"source": source},
                    )
                )

            state_value = env.state() if callable(getattr(env, "state", None)) else getattr(env, "state", None)
            score = float(getattr(state_value, "score", result.observation.metadata.get("score", 0.0)))
            success = bool(result.observation.grader_passed and result.done)
            if success:
                failure_type = None
                failure_reason = None
            elif len(steps) >= self.max_steps and not result.done:
                failure_type = "max_steps"
                failure_reason = "Max orchestrator step limit reached."
            elif result.done:
                failure_type = "submission_incomplete"
                failure_reason = result.observation.status_message
            else:
                failure_type = "unsolved"
                failure_reason = "Episode ended without a fully verified repair."

            return EpisodeResult(
                run_id=run_id,
                episode_id=f"{run_id}:{task.task_id}:{started_at}",
                task_id=task.task_id,
                difficulty=task.difficulty,
                policy_name=self.policy_name,
                model_name=self.model_name,
                success=success,
                score=score,
                step_count=len(steps),
                failure_type=failure_type,
                failure_reason=failure_reason,
                started_at=started_at,
                finished_at=utc_now(),
                steps=steps,
            )
        finally:
            env.__exit__(None, None, None)
