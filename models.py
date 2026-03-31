"""Typed models for the WebSec Repair environment."""

from __future__ import annotations

from typing import Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class PatchOption(BaseModel):
    """One available patch template for the active vulnerability task."""

    id: str
    label: str


class WebSecRepairAction(Action):
    """Actions available to an agent inside the environment."""

    action_type: Literal["inspect", "classify", "apply_patch", "verify", "submit"]
    vulnerability_type: str | None = Field(
        default=None,
        description="Chosen vulnerability label for classify actions.",
    )
    patch_id: str | None = Field(
        default=None,
        description="Chosen patch option for apply_patch actions.",
    )


class WebSecRepairObservation(Observation):
    """Observation returned after each environment step."""

    task_id: str = Field(default="", description="Active task id.")
    instruction: str = Field(default="", description="Task instruction.")
    code_snippet: str = Field(
        default="",
        description="Visible vulnerable code snippet after inspect.",
    )
    scanner_hint: str = Field(
        default="",
        description="Visible scanner hint after inspect.",
    )
    status_message: str = Field(
        default="",
        description="Human-readable result of the last action.",
    )
    workflow_stage: Literal["start", "inspected", "classified", "patched", "verified", "submitted"] = Field(
        default="start",
        description="Current workflow phase for the episode.",
    )
    selected_vulnerability: str = Field(
        default="",
        description="Currently selected vulnerability label.",
    )
    applied_patch_id: str = Field(
        default="",
        description="Currently applied patch option id.",
    )
    patch_options: list[PatchOption] = Field(
        default_factory=list,
        description="Patch options visible after inspect.",
    )
    exploit_test_passed: bool = Field(
        default=False,
        description="Whether the exploit is blocked.",
    )
    functionality_test_passed: bool = Field(
        default=False,
        description="Whether the legitimate behavior is preserved.",
    )
    grader_passed: bool = Field(
        default=False,
        description="Whether the task fully passes the grader.",
    )


class WebSecRepairState(State):
    """Persistent episode state tracked by the environment."""

    task_id: str = Field(default="sqli_login", description="Active task id.")
    difficulty: str = Field(default="easy", description="Difficulty bucket.")
    workflow_stage: Literal["start", "inspected", "classified", "patched", "verified", "submitted"] = Field(
        default="start",
        description="Highest valid workflow phase reached in this episode.",
    )
    inspected: bool = Field(
        default=False,
        description="Whether inspect has been called in this episode.",
    )
    selected_vulnerability: str = Field(
        default="",
        description="Chosen vulnerability classification.",
    )
    applied_patch_id: str = Field(default="", description="Chosen patch option id.")
    exploit_test_passed: bool = Field(
        default=False,
        description="Whether verify blocked the exploit.",
    )
    functionality_test_passed: bool = Field(
        default=False,
        description="Whether verify preserved functionality.",
    )
    submitted: bool = Field(default=False, description="Whether submit was called.")
    score: float = Field(default=0.0, description="Current absolute grader score.")


class TaskDefinition(BaseModel):
    """Public task metadata."""

    id: str
    difficulty: str
    title: str
    instruction: str
    code_snippet: str
    scanner_hint: str
    patch_options: list[PatchOption] = Field(default_factory=list)


class TaskCatalog(BaseModel):
    """Response model for /tasks."""

    environment: str
    default_task_id: str
    available_difficulties: list[str] = Field(default_factory=list)
    filtered_difficulty: str | None = None
    tasks: list[TaskDefinition]


class GraderCheck(BaseModel):
    """One individual grader check."""

    name: str
    passed: bool
    detail: str


class GraderReport(BaseModel):
    """Current grader state for a task."""

    task_id: str
    passed: bool
    score: float
    message: str
    checks: list[GraderCheck] = Field(default_factory=list)


class BaselineActionStep(BaseModel):
    """One baseline action in the reference trajectory."""

    action_type: Literal["inspect", "classify", "apply_patch", "verify", "submit"]
    vulnerability_type: str | None = None
    patch_id: str | None = None


class BaselineDefinition(BaseModel):
    """Public baseline trajectory for one task."""

    task_id: str
    title: str
    description: str
    actions: list[BaselineActionStep] = Field(default_factory=list)


class BaselineCatalog(BaseModel):
    """Response model for /baseline."""

    environment: str
    baselines: list[BaselineDefinition] = Field(default_factory=list)


class RuntimeStatus(BaseModel):
    """Live runtime progress and grader outcome for one task."""

    environment: str
    progress: WebSecRepairState
    grader: GraderReport
