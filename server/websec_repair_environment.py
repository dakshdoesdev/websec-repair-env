"""Environment implementation for deterministic web security repair tasks."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..models import WebSecRepairAction, WebSecRepairObservation, WebSecRepairState
    from .challenge import (
        DEFAULT_TASK_ID,
        TASKS,
        get_task,
        grade_task,
        reset_runtime_progress,
        set_runtime_progress,
        verification_outcome,
    )
except ImportError:
    from models import WebSecRepairAction, WebSecRepairObservation, WebSecRepairState  # type: ignore
    from server.challenge import (  # type: ignore
        DEFAULT_TASK_ID,
        TASKS,
        get_task,
        grade_task,
        reset_runtime_progress,
        set_runtime_progress,
        verification_outcome,
    )

MAX_STEPS = 6
STAGE_ORDER = ("start", "inspected", "classified", "patched", "verified", "submitted")
REWARD_VALUES = {
    "inspect_first": 0.05,
    "classify_correct": 0.20,
    "patch_correct": 0.30,
    "verify_exploit": 0.15,
    "verify_functionality": 0.15,
    "verify_bonus": 0.05,
    "submit_success": 0.10,
}
PENALTY_VALUES = {
    "repeat_inspect": -0.02,
    "repeat_classify": -0.02,
    "missing_argument": -0.08,
    "classify_before_inspect": -0.08,
    "classify_wrong": -0.10,
    "patch_before_classify": -0.15,
    "patch_unknown": -0.10,
    "patch_wrong": -0.18,
    "repeat_patch": -0.02,
    "verify_before_patch": -0.12,
    "repeat_verify": -0.02,
    "verify_failed_check": -0.10,
    "submit_incomplete": -0.20,
    "post_submit_action": -0.05,
}


class WebSecRepairEnvironment(
    Environment[WebSecRepairAction, WebSecRepairObservation, WebSecRepairState]
):
    """Lean deterministic environment for vulnerability classification and repair."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self) -> None:
        super().__init__()
        default_task = get_task(DEFAULT_TASK_ID)
        self._state = WebSecRepairState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=default_task.id,
            difficulty=default_task.difficulty,
            workflow_stage="start",
            inspected=False,
            selected_vulnerability="",
            applied_patch_id="",
            exploit_test_passed=False,
            functionality_test_passed=False,
            submitted=False,
            score=0.0,
        )
        set_runtime_progress(self._state.model_dump())

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> WebSecRepairObservation:
        del seed

        task_id = kwargs.get("task_id", DEFAULT_TASK_ID)
        task = get_task(task_id)
        reset_runtime_progress(task_id)

        self._state = WebSecRepairState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task.id,
            difficulty=task.difficulty,
            workflow_stage="start",
            inspected=False,
            selected_vulnerability="",
            applied_patch_id="",
            exploit_test_passed=False,
            functionality_test_passed=False,
            submitted=False,
            score=0.0,
        )
        set_runtime_progress(self._state.model_dump())

        return self._build_observation(
            status_message="Task loaded. Use inspect to reveal the snippet, hint, and patch options.",
            reward=0.0,
            done=False,
        )

    def step(
        self,
        action: WebSecRepairAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> WebSecRepairObservation:
        del timeout_s, kwargs

        self._state.step_count += 1
        reward = 0.0
        status_message = ""
        task_runtime = TASKS[self._state.task_id]
        task_definition = task_runtime["definition"]
        correct_vulnerability = task_runtime["correct_vulnerability"]
        correct_patch = task_runtime["correct_patch"]

        if self._state.submitted:
            reward += PENALTY_VALUES["post_submit_action"]
            status_message = "Episode already submitted. Reset before taking more actions."
        elif action.action_type == "inspect":
            if self._state.inspected:
                reward += PENALTY_VALUES["repeat_inspect"]
                status_message = "Inspection repeated. No new information was revealed."
            else:
                self._state.inspected = True
                self._state.workflow_stage = "inspected"
                reward += REWARD_VALUES["inspect_first"]
                status_message = "Inspection complete. The vulnerable snippet, scanner hint, and patch options are now visible."
        elif action.action_type == "classify":
            if not self._state.inspected:
                reward += PENALTY_VALUES["classify_before_inspect"]
                status_message = "Inspect first. Classification guesses before inspection are penalized."
            elif not action.vulnerability_type:
                reward += PENALTY_VALUES["missing_argument"]
                status_message = "Classification failed: vulnerability_type is required."
            else:
                self._state.selected_vulnerability = action.vulnerability_type
                self._state.applied_patch_id = ""
                self._clear_verification_state()
                if action.vulnerability_type == correct_vulnerability:
                    if self._state.workflow_stage == "classified":
                        reward += PENALTY_VALUES["repeat_classify"]
                        status_message = f"Classification {action.vulnerability_type!r} was already correct."
                    else:
                        self._state.workflow_stage = "classified"
                        reward += REWARD_VALUES["classify_correct"]
                        status_message = f"Stored correct vulnerability classification {action.vulnerability_type!r}."
                else:
                    self._state.workflow_stage = "inspected"
                    reward += PENALTY_VALUES["classify_wrong"]
                    status_message = (
                        f"Stored vulnerability classification {action.vulnerability_type!r}, "
                        "but it does not match the task."
                    )
        elif action.action_type == "apply_patch":
            if not self._state.inspected or self._state.selected_vulnerability != correct_vulnerability:
                reward += PENALTY_VALUES["patch_before_classify"]
                status_message = "Patch selection requires a correct classification after inspection."
            elif not action.patch_id:
                reward += PENALTY_VALUES["missing_argument"]
                status_message = "Patch application failed: patch_id is required."
            else:
                valid_patch_ids = {option.id for option in task_definition.patch_options}
                if action.patch_id not in valid_patch_ids:
                    reward += PENALTY_VALUES["patch_unknown"]
                    status_message = f"Patch application failed: unknown patch_id {action.patch_id!r}."
                else:
                    if self._state.applied_patch_id == action.patch_id and self._state.workflow_stage in {"patched", "verified"}:
                        reward += PENALTY_VALUES["repeat_patch"]
                        self._state.workflow_stage = "patched"
                        self._clear_verification_state()
                        status_message = f"Patch template {action.patch_id!r} is already applied."
                    else:
                        self._state.workflow_stage = "patched"
                        self._clear_verification_state()
                        self._state.applied_patch_id = action.patch_id
                        if action.patch_id == correct_patch:
                            reward += REWARD_VALUES["patch_correct"]
                            status_message = f"Applied patch template {action.patch_id!r}. Verification is now required."
                        else:
                            reward += PENALTY_VALUES["patch_wrong"]
                            status_message = (
                                f"Applied patch template {action.patch_id!r}, "
                                "but it is not the safest repair for this task."
                            )
        elif action.action_type == "verify":
            if not self._state.inspected or not self._state.applied_patch_id:
                reward += PENALTY_VALUES["verify_before_patch"]
                status_message = "Verify requires an inspected task and an applied patch."
            elif self._state.selected_vulnerability != correct_vulnerability:
                reward += PENALTY_VALUES["verify_before_patch"]
                status_message = "Verify is penalized until the vulnerability is classified correctly."
            elif self._state.workflow_stage == "verified":
                reward += PENALTY_VALUES["repeat_verify"]
                status_message = "Verification already passed for the current patch."
            else:
                exploit, functionality, verify_message = verification_outcome(
                    self._state.task_id,
                    self._state.applied_patch_id or None,
                )
                self._state.exploit_test_passed = exploit
                self._state.functionality_test_passed = functionality
                reward += REWARD_VALUES["verify_exploit"] if exploit else PENALTY_VALUES["verify_failed_check"]
                reward += REWARD_VALUES["verify_functionality"] if functionality else PENALTY_VALUES["verify_failed_check"]
                if exploit and functionality:
                    self._state.workflow_stage = "verified"
                    reward += REWARD_VALUES["verify_bonus"]
                else:
                    self._state.workflow_stage = "patched"
                status_message = verify_message
        elif action.action_type == "submit":
            self._state.submitted = True
            solved = (
                self._state.inspected
                and self._state.selected_vulnerability == correct_vulnerability
                and self._state.applied_patch_id == correct_patch
                and self._state.exploit_test_passed
                and self._state.functionality_test_passed
            )
            if solved:
                self._state.workflow_stage = "submitted"
                reward += REWARD_VALUES["submit_success"]
                status_message = "Submission recorded. Task solved."
            else:
                reward += PENALTY_VALUES["submit_incomplete"]
                status_message = "Submission recorded before a fully verified repair. Episode failed."
        else:
            reward += PENALTY_VALUES["missing_argument"]
            status_message = f"Unsupported action_type {action.action_type!r}."

        reward = round(reward, 2)
        self._state.score = round(self._state.score + reward, 2)
        report = grade_task(progress=self._state.model_dump())
        set_runtime_progress(self._state.model_dump())

        done = self._state.submitted or self._state.step_count >= MAX_STEPS
        if self._state.step_count >= MAX_STEPS and not self._state.submitted:
            status_message = f"{status_message} Max steps reached."

        return self._build_observation(
            status_message=status_message.strip(),
            reward=reward,
            done=done,
            grader_report=report,
        )

    def _clear_verification_state(self) -> None:
        self._state.exploit_test_passed = False
        self._state.functionality_test_passed = False

    def _recommended_next_action(self) -> str:
        task_runtime = TASKS[self._state.task_id]
        if self._state.submitted:
            return "reset"
        if not self._state.inspected:
            return "inspect"
        if self._state.selected_vulnerability != task_runtime["correct_vulnerability"]:
            return "classify"
        if not self._state.applied_patch_id:
            return "apply_patch"
        if not (self._state.exploit_test_passed and self._state.functionality_test_passed):
            return "verify"
        return "submit"

    @property
    def state(self) -> WebSecRepairState:
        """Return the current environment state."""
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        """Return environment metadata for the OpenEnv UI."""
        return EnvironmentMetadata(
            name="WebSecRepairEnvironment",
            description=(
                "Deterministic OpenEnv environment with three web vulnerability repair tasks "
                "covering SQL injection, XSS, and broken access control."
            ),
            version="0.1.0",
            author="Codex",
        )

    def _build_observation(
        self,
        status_message: str,
        reward: float,
        done: bool,
        grader_report: Any | None = None,
    ) -> WebSecRepairObservation:
        task = get_task(self._state.task_id)
        if grader_report is None:
            grader_report = grade_task(progress=self._state.model_dump())
        visible_patch_options = task.patch_options if self._state.inspected else []
        visible_code = task.code_snippet if self._state.inspected else ""
        visible_hint = task.scanner_hint if self._state.inspected else ""
        return WebSecRepairObservation(
            task_id=task.id,
            instruction=task.instruction,
            code_snippet=visible_code,
            scanner_hint=visible_hint,
            status_message=status_message,
            workflow_stage=self._state.workflow_stage,
            selected_vulnerability=self._state.selected_vulnerability,
            applied_patch_id=self._state.applied_patch_id,
            patch_options=visible_patch_options,
            exploit_test_passed=self._state.exploit_test_passed,
            functionality_test_passed=self._state.functionality_test_passed,
            grader_passed=grader_report.passed,
            reward=reward,
            done=done,
            metadata={
                "step_count": self._state.step_count,
                "score": self._state.score,
                "grader_score": grader_report.score,
                "max_steps": MAX_STEPS,
                "workflow_stage": self._state.workflow_stage,
                "recommended_next_action": self._recommended_next_action(),
                "grader": grader_report.model_dump(),
            },
        )
