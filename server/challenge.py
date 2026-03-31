"""Task definitions, baseline trajectories, and grader logic."""

from __future__ import annotations

from copy import deepcopy

try:
    from ..models import (
        BaselineActionStep,
        BaselineCatalog,
        BaselineDefinition,
        GraderCheck,
        GraderReport,
        PatchOption,
        TaskCatalog,
        TaskDefinition,
    )
except ImportError:
    from models import (  # type: ignore
        BaselineActionStep,
        BaselineCatalog,
        BaselineDefinition,
        GraderCheck,
        GraderReport,
        PatchOption,
        TaskCatalog,
        TaskDefinition,
    )

ENVIRONMENT_NAME = "websec_repair_env"
DEFAULT_TASK_ID = "sqli_login"

_PATCH_WEIGHTS = {
    "classification": 0.25,
    "patch": 0.35,
    "exploit": 0.20,
    "functionality": 0.15,
    "submit": 0.05,
}

TASKS = {
    "sqli_login": {
        "definition": TaskDefinition(
            id="sqli_login",
            difficulty="easy",
            title="Patch the vulnerable login query",
            instruction=(
                "Inspect the login handler, classify the vulnerability, choose one patch, "
                "verify exploit blocking and functionality preservation, then submit."
            ),
            code_snippet=(
                "def login(username, password):\n"
                "    query = (\n"
                "        \"SELECT id FROM users WHERE username = '\" + username + \"' \"\n"
                "        \"AND password = '\" + password + \"'\"\n"
                "    )\n"
                "    return db.execute(query).fetchone()\n"
            ),
            scanner_hint=(
                "Scanner warning: SQL injection risk detected in login query due to "
                "string concatenation with user-controlled input."
            ),
            patch_options=[
                PatchOption(id="parameterized_query", label="Use parameterized query"),
                PatchOption(id="strip_quotes", label="Strip quotes from input"),
                PatchOption(id="disable_login", label="Disable login route"),
            ],
        ),
        "correct_vulnerability": "sql_injection",
        "correct_patch": "parameterized_query",
        "verify": {
            "parameterized_query": (
                True,
                True,
                "Exploit is blocked and valid logins still work with bound parameters.",
            ),
            "strip_quotes": (
                False,
                True,
                "Input filtering is incomplete; the injection path is still reachable.",
            ),
            "disable_login": (
                True,
                False,
                "Attack path is gone because login is disabled, but legitimate users are locked out.",
            ),
        },
    },
    "xss_comments": {
        "definition": TaskDefinition(
            id="xss_comments",
            difficulty="medium",
            title="Stop script execution in comment rendering",
            instruction=(
                "Inspect the comment renderer, classify the vulnerability, choose one patch, "
                "verify exploit blocking and functionality preservation, then submit."
            ),
            code_snippet=(
                "def render_comment(comment):\n"
                '    return f"<div class=\\"comment\\">{comment}</div>"\n'
            ),
            scanner_hint=(
                "Scanner warning: Cross-site scripting risk detected because comment text is "
                "rendered into HTML without escaping."
            ),
            patch_options=[
                PatchOption(id="html_escape", label="Escape comment before rendering"),
                PatchOption(id="remove_script_substring", label="Remove 'script' substring"),
                PatchOption(id="disable_comment_rendering", label="Hide comments completely"),
            ],
        ),
        "correct_vulnerability": "xss",
        "correct_patch": "html_escape",
        "verify": {
            "html_escape": (
                True,
                True,
                "Malicious payloads render as text and normal comments still display.",
            ),
            "remove_script_substring": (
                False,
                True,
                "Substring filtering misses alternate payloads; comments still render.",
            ),
            "disable_comment_rendering": (
                True,
                False,
                "No script runs because comments are removed, but legitimate content is gone too.",
            ),
        },
    },
    "broken_auth_admin": {
        "definition": TaskDefinition(
            id="broken_auth_admin",
            difficulty="hard",
            title="Restore admin-only access control",
            instruction=(
                "Inspect the admin route, classify the vulnerability, choose one patch, "
                "verify exploit blocking and functionality preservation, then submit."
            ),
            code_snippet=(
                "@app.get('/admin')\n"
                "def admin_panel(current_user):\n"
                "    if not current_user:\n"
                "        raise UnauthorizedError()\n"
                "    return render_admin_dashboard()\n"
            ),
            scanner_hint=(
                "Scanner warning: Broken access control. Authenticated non-admin users can reach "
                "the admin route because role enforcement is missing."
            ),
            patch_options=[
                PatchOption(id="require_admin_role", label="Enforce admin role server-side"),
                PatchOption(id="hide_admin_link", label="Hide the admin link in UI"),
                PatchOption(id="deny_all_admin_access", label="Block all admin route traffic"),
            ],
        ),
        "correct_vulnerability": "broken_auth",
        "correct_patch": "require_admin_role",
        "verify": {
            "require_admin_role": (
                True,
                True,
                "Non-admin users are denied while legitimate admins still reach the dashboard.",
            ),
            "hide_admin_link": (
                False,
                True,
                "The UI hides the link, but direct requests still bypass authorization.",
            ),
            "deny_all_admin_access": (
                True,
                False,
                "The bypass is gone, but valid admins are blocked too.",
            ),
        },
    },
}

BASELINES = {
    task_id: BaselineDefinition(
        task_id=task_id,
        title=f"{task['definition'].title} baseline",
        description="Inspect, classify correctly, apply the safe patch, verify, then submit.",
        actions=[
            BaselineActionStep(action_type="inspect"),
            BaselineActionStep(
                action_type="classify",
                vulnerability_type=task["correct_vulnerability"],
            ),
            BaselineActionStep(
                action_type="apply_patch",
                patch_id=task["correct_patch"],
            ),
            BaselineActionStep(action_type="verify"),
            BaselineActionStep(action_type="submit"),
        ],
    )
    for task_id, task in TASKS.items()
}


def _blank_progress(task_id: str) -> dict[str, object]:
    try:
        task = TASKS[task_id]["definition"]
    except KeyError as exc:
        valid = ", ".join(sorted(TASKS))
        raise ValueError(f"Unknown task_id {task_id!r}. Expected one of: {valid}") from exc
    return {
        "task_id": task.id,
        "difficulty": task.difficulty,
        "workflow_stage": "start",
        "inspected": False,
        "selected_vulnerability": "",
        "applied_patch_id": "",
        "exploit_test_passed": False,
        "functionality_test_passed": False,
        "submitted": False,
        "score": 0.0,
    }


_CURRENT_PROGRESS = _blank_progress(DEFAULT_TASK_ID)


def get_task(task_id: str) -> TaskDefinition:
    """Resolve one public task definition."""
    try:
        return TASKS[task_id]["definition"]
    except KeyError as exc:
        valid = ", ".join(sorted(TASKS))
        raise ValueError(f"Unknown task_id {task_id!r}. Expected one of: {valid}") from exc


def list_difficulties() -> list[str]:
    """Return all difficulty levels present in the task catalog."""
    return sorted({task["definition"].difficulty for task in TASKS.values()})


def list_tasks(difficulty: str | None = None) -> TaskCatalog:
    """Return the task catalog, optionally filtered by difficulty."""
    all_tasks = [TASKS["sqli_login"]["definition"], TASKS["xss_comments"]["definition"], TASKS["broken_auth_admin"]["definition"]]
    difficulties = list_difficulties()
    if difficulty is not None and difficulty not in difficulties:
        valid = ", ".join(difficulties)
        raise ValueError(f"Unknown difficulty {difficulty!r}. Expected one of: {valid}")
    filtered_tasks = all_tasks if difficulty is None else [task for task in all_tasks if task.difficulty == difficulty]
    return TaskCatalog(
        environment=ENVIRONMENT_NAME,
        default_task_id=DEFAULT_TASK_ID,
        available_difficulties=difficulties,
        filtered_difficulty=difficulty,
        tasks=filtered_tasks,
    )


def list_baselines(task_id: str | None = None) -> BaselineCatalog:
    """Return all baseline trajectories or one filtered trajectory."""
    if task_id is None:
        baselines = [BASELINES["sqli_login"], BASELINES["xss_comments"], BASELINES["broken_auth_admin"]]
    else:
        get_task(task_id)
        baselines = [BASELINES[task_id]]
    return BaselineCatalog(environment=ENVIRONMENT_NAME, baselines=baselines)


def get_baseline(task_id: str) -> BaselineDefinition:
    """Return one baseline trajectory."""
    get_task(task_id)
    return BASELINES[task_id]


def reset_runtime_progress(task_id: str) -> dict[str, object]:
    """Reset the shared runtime snapshot for the requested task."""
    global _CURRENT_PROGRESS
    _CURRENT_PROGRESS = _blank_progress(task_id)
    return deepcopy(_CURRENT_PROGRESS)


def set_runtime_progress(progress: dict[str, object]) -> dict[str, object]:
    """Replace the shared runtime snapshot with the provided progress."""
    global _CURRENT_PROGRESS
    task_id = str(progress.get("task_id", DEFAULT_TASK_ID))
    get_task(task_id)
    merged = _blank_progress(task_id)
    merged.update(progress)
    _CURRENT_PROGRESS = merged
    return deepcopy(_CURRENT_PROGRESS)


def current_runtime_progress() -> dict[str, object]:
    """Return the current shared runtime snapshot."""
    return deepcopy(_CURRENT_PROGRESS)


def verification_outcome(task_id: str, patch_id: str | None) -> tuple[bool, bool, str]:
    """Evaluate exploit and functionality checks for the selected patch."""
    get_task(task_id)
    if not patch_id:
        return False, False, "No patch has been applied yet, so both checks fail."
    try:
        return TASKS[task_id]["verify"][patch_id]
    except KeyError as exc:
        valid = ", ".join(option.id for option in TASKS[task_id]["definition"].patch_options)
        raise ValueError(f"Unknown patch_id {patch_id!r}. Expected one of: {valid}") from exc


def grade_task(
    task_id: str | None = None,
    progress: dict[str, object] | None = None,
) -> GraderReport:
    """Grade either the provided progress snapshot or the shared runtime state."""
    if progress is None:
        if task_id is None:
            working = current_runtime_progress()
        else:
            get_task(task_id)
            working = current_runtime_progress()
            if working["task_id"] != task_id:
                working = _blank_progress(task_id)
    else:
        working = deepcopy(progress)
        task_id = str(working.get("task_id", task_id or DEFAULT_TASK_ID))
        get_task(task_id)

    resolved_task_id = str(task_id or working["task_id"])
    task = TASKS[resolved_task_id]

    classification_correct = working.get("selected_vulnerability") == task["correct_vulnerability"]
    patch_correct = working.get("applied_patch_id") == task["correct_patch"]
    exploit_blocked = bool(working.get("exploit_test_passed", False))
    functionality_preserved = bool(working.get("functionality_test_passed", False))
    successful_submit = (
        bool(working.get("submitted", False))
        and classification_correct
        and patch_correct
        and exploit_blocked
        and functionality_preserved
    )

    score = 0.0
    if classification_correct:
        score += _PATCH_WEIGHTS["classification"]
    if patch_correct:
        score += _PATCH_WEIGHTS["patch"]
    if exploit_blocked:
        score += _PATCH_WEIGHTS["exploit"]
    if functionality_preserved:
        score += _PATCH_WEIGHTS["functionality"]
    if successful_submit:
        score += _PATCH_WEIGHTS["submit"]

    checks = [
        GraderCheck(
            name="classification_correct",
            passed=classification_correct,
            detail=(
                f"Expected {task['correct_vulnerability']!r}; "
                f"received {working.get('selected_vulnerability', '')!r}."
            ),
        ),
        GraderCheck(
            name="patch_correct",
            passed=patch_correct,
            detail=(
                f"Expected {task['correct_patch']!r}; "
                f"received {working.get('applied_patch_id', '')!r}."
            ),
        ),
        GraderCheck(
            name="exploit_blocked",
            passed=exploit_blocked,
            detail=(
                "Verify step confirms the exploit is blocked."
                if exploit_blocked
                else "Exploit is still possible or verify has not been run."
            ),
        ),
        GraderCheck(
            name="functionality_preserved",
            passed=functionality_preserved,
            detail=(
                "Verify step confirms legitimate behavior still works."
                if functionality_preserved
                else "Legitimate behavior is broken or verify has not been run."
            ),
        ),
        GraderCheck(
            name="submitted",
            passed=successful_submit,
            detail=(
                "Task was submitted after all required checks passed."
                if successful_submit
                else "Submit bonus only applies after the correct verified repair is submitted."
            ),
        ),
    ]

    passed = all(check.passed for check in checks)
    message = "Task solved." if passed else "Task is not solved yet."
    return GraderReport(
        task_id=resolved_task_id,
        passed=passed,
        score=round(score, 2),
        message=message,
        checks=checks,
    )
