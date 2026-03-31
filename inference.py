#!/usr/bin/env python3
"""Deterministic baseline agent for WebSec Repair Env."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from websec_repair_env import WebSecRepairAction, WebSecRepairEnv


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


def choose_vulnerability(task_id: str, scanner_hint: str) -> str:
    """Pick the deterministic vulnerability label for the baseline."""
    lowered = scanner_hint.lower()
    for keyword, label in HINT_KEYWORDS.items():
        if keyword in lowered:
            return label
    return TASK_TO_VULNERABILITY[task_id]


def run_baseline(base_url: str, task_id: str) -> int:
    """Run the deterministic baseline policy against a running environment."""
    with WebSecRepairEnv(base_url=base_url).sync() as env:
        result = env.reset(task_id=task_id)
        print(f"reset: task={result.observation.task_id}")

        result = env.step(WebSecRepairAction(action_type="inspect"))
        print(f"inspect: reward={result.reward} status={result.observation.status_message}")

        vulnerability = choose_vulnerability(
            result.observation.task_id,
            result.observation.scanner_hint,
        )
        result = env.step(
            WebSecRepairAction(
                action_type="classify",
                vulnerability_type=vulnerability,
            )
        )
        print(f"classify: reward={result.reward} selected={result.observation.selected_vulnerability}")

        patch_id = TASK_TO_PATCH[result.observation.task_id]
        result = env.step(
            WebSecRepairAction(
                action_type="apply_patch",
                patch_id=patch_id,
            )
        )
        print(f"apply_patch: reward={result.reward} patch={result.observation.applied_patch_id}")

        result = env.step(WebSecRepairAction(action_type="verify"))
        print(
            "verify: "
            f"reward={result.reward} exploit={result.observation.exploit_test_passed} "
            f"functionality={result.observation.functionality_test_passed}"
        )

        result = env.step(WebSecRepairAction(action_type="submit"))
        print(
            "submit: "
            f"reward={result.reward} done={result.done} passed={result.observation.grader_passed} "
            f"score={result.observation.reward}"
        )
        return 0 if result.observation.grader_passed else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--task",
        default="sqli_login",
        choices=sorted(TASK_TO_VULNERABILITY),
    )
    args = parser.parse_args()
    raise SystemExit(run_baseline(args.base_url, args.task))


if __name__ == "__main__":
    main()
