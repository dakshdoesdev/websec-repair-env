"""Typed client for the WebSec Repair environment."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import PatchOption, WebSecRepairAction, WebSecRepairObservation, WebSecRepairState


class WebSecRepairEnv(
    EnvClient[WebSecRepairAction, WebSecRepairObservation, WebSecRepairState]
):
    """WebSocket client for the deterministic web security repair environment."""

    def _step_payload(self, action: WebSecRepairAction) -> Dict[str, Any]:
        return {
            "action_type": action.action_type,
            "vulnerability_type": action.vulnerability_type,
            "patch_id": action.patch_id,
            "metadata": action.metadata,
        }

    def _parse_result(
        self,
        payload: Dict[str, Any],
    ) -> StepResult[WebSecRepairObservation]:
        obs_data = payload.get("observation", {})
        observation = WebSecRepairObservation(
            task_id=obs_data.get("task_id", ""),
            instruction=obs_data.get("instruction", ""),
            code_snippet=obs_data.get("code_snippet", ""),
            scanner_hint=obs_data.get("scanner_hint", ""),
            status_message=obs_data.get("status_message", ""),
            workflow_stage=obs_data.get("workflow_stage", "start"),
            selected_vulnerability=obs_data.get("selected_vulnerability", ""),
            applied_patch_id=obs_data.get("applied_patch_id", ""),
            patch_options=[PatchOption(**item) for item in obs_data.get("patch_options", [])],
            exploit_test_passed=obs_data.get("exploit_test_passed", False),
            functionality_test_passed=obs_data.get("functionality_test_passed", False),
            grader_passed=obs_data.get("grader_passed", False),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> WebSecRepairState:
        return WebSecRepairState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", "sqli_login"),
            difficulty=payload.get("difficulty", "easy"),
            workflow_stage=payload.get("workflow_stage", "start"),
            inspected=payload.get("inspected", False),
            selected_vulnerability=payload.get("selected_vulnerability", ""),
            applied_patch_id=payload.get("applied_patch_id", ""),
            exploit_test_passed=payload.get("exploit_test_passed", False),
            functionality_test_passed=payload.get("functionality_test_passed", False),
            submitted=payload.get("submitted", False),
            score=payload.get("score", 0.0),
        )
