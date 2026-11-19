"""Shared local-LLM baseline policy for the WebSec Repair environment."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .models import WebSecRepairAction, WebSecRepairObservation
from .training.inference import extract_env_action, parse_model_output

DEFAULT_API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:11434/v1")
DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "qwen2.5:1.5b")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "local")

TASK_TO_VULNERABILITY = {
    "sqli_login": "sql_injection",
    "xss_comments": "xss",
    "broken_auth_admin": "broken_auth",
}
TASK_TO_PATCH = {
    "sqli_login": "parameterized_query",
    "xss_comments": "html_escape",
    "broken_auth_admin": "require_admin_role",
}
HINT_KEYWORDS = {
    "sql injection": "sql_injection",
    "cross-site scripting": "xss",
    "xss": "xss",
    "access control": "broken_auth",
    "authorization": "broken_auth",
    "admin route": "broken_auth",
}

SYSTEM_PROMPT = """You are a web security repair assistant.

Choose exactly one next environment action and return JSON only.

Allowed actions:
{"action_type":"inspect"}
{"action_type":"classify","vulnerability_type":"sql_injection"}
{"action_type":"classify","vulnerability_type":"xss"}
{"action_type":"classify","vulnerability_type":"broken_auth"}
{"action_type":"apply_patch","patch_id":"parameterized_query"}
{"action_type":"apply_patch","patch_id":"html_escape"}
{"action_type":"apply_patch","patch_id":"require_admin_role"}
{"action_type":"verify"}
{"action_type":"submit"}

Rules:
- If evidence is hidden, inspect first.
- After inspect, classify the vulnerability from the code and scanner hint.
- Only choose a patch after correct classification.
- Verify before submit.
- Prefer the safest server-side fix, not UI-only or destructive fixes.
"""


@dataclass(frozen=True)
class PolicyConfig:
    api_base_url: str
    model_name: str
    api_key: str


def build_policy_config(
    *,
    api_base_url: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
) -> PolicyConfig:
    return PolicyConfig(
        api_base_url=api_base_url or DEFAULT_API_BASE_URL,
        model_name=model_name or DEFAULT_MODEL_NAME,
        api_key=api_key or DEFAULT_API_KEY,
    )


def choose_vulnerability(task_id: str, scanner_hint: str) -> str:
    lowered = scanner_hint.lower()
    for keyword, label in HINT_KEYWORDS.items():
        if keyword in lowered:
            return label
    return TASK_TO_VULNERABILITY[task_id]


def fallback_action(
    observation: WebSecRepairObservation,
    history: list[dict[str, Any]],
) -> WebSecRepairAction:
    if not observation.code_snippet or not observation.scanner_hint:
        return WebSecRepairAction(action_type="inspect")
    if not observation.selected_vulnerability:
        return WebSecRepairAction(
            action_type="classify",
            vulnerability_type=choose_vulnerability(observation.task_id, observation.scanner_hint),
        )
    if not observation.applied_patch_id:
        return WebSecRepairAction(
            action_type="apply_patch",
            patch_id=TASK_TO_PATCH[observation.task_id],
        )
    if not (observation.exploit_test_passed and observation.functionality_test_passed):
        return WebSecRepairAction(action_type="verify")
    return WebSecRepairAction(action_type="submit")


def apply_guardrails(
    observation: WebSecRepairObservation,
    action: WebSecRepairAction,
) -> tuple[WebSecRepairAction, str | None]:
    if observation.exploit_test_passed and observation.functionality_test_passed:
        if action.action_type != "submit":
            return WebSecRepairAction(action_type="submit"), "post-verify submit guardrail"

    if not observation.code_snippet and action.action_type != "inspect":
        return WebSecRepairAction(action_type="inspect"), "inspect-first guardrail"

    if observation.code_snippet and not observation.selected_vulnerability and action.action_type == "apply_patch":
        return WebSecRepairAction(
            action_type="classify",
            vulnerability_type=choose_vulnerability(observation.task_id, observation.scanner_hint),
        ), "classify-before-patch guardrail"

    return action, None


def _history_text(history: list[dict[str, Any]]) -> str:
    if not history:
        return "No previous steps."
    lines = []
    for item in history[-4:]:
        lines.append(
            f"- step {item.get('step')}: action={item.get('action')} reward={item.get('reward')} result={item.get('status_message')}"
        )
    return "\n".join(lines)


def _build_user_prompt(observation: WebSecRepairObservation, history: list[dict[str, Any]]) -> str:
    patch_options = ", ".join(option.id for option in observation.patch_options) or "(hidden until inspect)"
    return f"""Task: {observation.task_id}
Instruction: {observation.instruction}
Workflow stage: {observation.workflow_stage}
Selected vulnerability: {observation.selected_vulnerability or '-'}
Applied patch: {observation.applied_patch_id or '-'}
Exploit test passed: {observation.exploit_test_passed}
Functionality test passed: {observation.functionality_test_passed}

Scanner hint:
{observation.scanner_hint or '(hidden until inspect)'}

Code snippet:
{observation.code_snippet or '(hidden until inspect)'}

Patch options:
{patch_options}

Recent history:
{_history_text(history)}

Return exactly one JSON object for the next action.
"""


class LocalLLMPolicy:
    def __init__(self, config: PolicyConfig) -> None:
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.api_base_url)

    def choose_action(
        self,
        observation: WebSecRepairObservation,
        history: list[dict[str, Any]],
    ) -> tuple[WebSecRepairAction, str, str | None]:
        fallback = fallback_action(observation, history)
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(observation, history)},
                ],
                temperature=0.0,
            )
            text = response.choices[0].message.content or ""
            payload = extract_env_action(parse_model_output(text))
            action, guardrail = apply_guardrails(observation, WebSecRepairAction(**payload))
            source = "local-llm"
            if guardrail:
                source = f"{source}+guardrail"
            return action, source, guardrail
        except Exception as exc:  # pragma: no cover - runtime fallback
            return fallback, "fallback", str(exc)


def format_action_payload(action: WebSecRepairAction) -> str:
    payload = action.model_dump(exclude_none=True)
    if payload.get("metadata") == {}:
        payload.pop("metadata", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
