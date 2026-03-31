"""Data loading and preprocessing for WebSec repair model training.

The training pipeline uses short, task-grounded supervision examples rather than
feeding long research prose directly into SFT. This module converts:

- deterministic trajectory logs from the environment
- synthetic canonical WebSec tasks
- a markdown/plain-text security research writeup
- optional SWE-bench Lite examples

into a consistent instruction/response dataset where the response follows a
dual-output JSON format:

{
  "action": {
    "action_type": "...",
    "vulnerability_type": "...",
    "patch_id": "..."
  },
  "review": {
    "owasp_top10_2021": "...",
    "cwe_id": "...",
    "cvss_v3_1": {"base_score": 0.0, "vector": "..."},
    "fix_family": "...",
    "verification_plan": ["..."],
    "rationale": "..."
  }
}
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, concatenate_datasets


DATASET_COLUMNS = [
    "instruction",
    "response",
    "task_id",
    "action_type",
    "source",
    "example_family",
]

ACTION_TYPES = {"inspect", "classify", "apply_patch", "verify", "submit"}
JSONL_SOURCE = "websec_jsonl"
VULNERABILITY_LABEL_ALIASES = {
    "auth_bypass": "broken_auth",
    "authentication_bypass": "broken_auth",
    "hardcoded_secret": "insecure_authentication",
    "hardcoded_credentials": "insecure_authentication",
    "weak_password_hashing": "insecure_authentication",
    "idor": "broken_access_control",
    "directory_traversal": "path_traversal",
    "os_command_injection": "command_injection",
    "rfi": "file_inclusion",
    "lfi": "file_inclusion",
}
SOURCE_GROUPS = {
    "custom": {JSONL_SOURCE, "trajectory", "synthetic", "research_doc", "synthetic_web_patterns"},
    "primevul_devign": {"primevul", "devign"},
    "juliet_owasp": {"juliet", "owasp_benchmark"},
    "patch_pairs": {"bigvul", "cvefixes", "github_security_fix", "nvd_cve", "exploitdb_selected"},
}
SOURCE_GROUP_TARGET_SHARES = {
    "custom": 0.35,
    "primevul_devign": 0.275,
    "juliet_owasp": 0.225,
    "patch_pairs": 0.15,
}
REPORT_FAMILIES = (
    "taxonomy_mapping",
    "patch_choice",
    "verification_planning",
    "workflow_policy",
)

RESPONSE_SCHEMA_TEXT = """Respond with JSON only using this schema:
{
  "action": {
    "action_type": "inspect | classify | apply_patch | verify | submit",
    "vulnerability_type": "optional string",
    "patch_id": "optional string"
  },
  "review": {
    "owasp_top10_2021": "optional string",
    "cwe_id": "optional string",
    "cvss_v3_1": {
      "base_score": 0.0,
      "vector": "optional string"
    },
    "fix_family": "optional string",
    "verification_plan": ["optional", "string", "steps"],
    "rationale": "short explanation"
  }
}
Omit optional fields when they are not needed."""


def _empty_dataset() -> Dataset:
    return Dataset.from_dict({column: [] for column in DATASET_COLUMNS})


def _make_dataset(examples: list[dict[str, Any]]) -> Dataset:
    if not examples:
        return _empty_dataset()

    normalized: dict[str, list[Any]] = {column: [] for column in DATASET_COLUMNS}
    for example in examples:
        for column in DATASET_COLUMNS:
            normalized[column].append(example.get(column, ""))
    return Dataset.from_dict(normalized)


def _concat_datasets(datasets: list[Dataset]) -> Dataset:
    if not datasets:
        return _empty_dataset()
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def _format_patch_options(patch_options: Iterable[dict[str, str]] | None) -> str:
    options = list(patch_options or [])
    if not options:
        return ""
    rendered = ["Patch options:"]
    for option in options:
        rendered.append(f"- {option['id']}: {option['label']}")
    return "\n".join(rendered)


def _infer_example_family(action_type: str) -> str:
    if action_type == "classify":
        return "taxonomy_mapping"
    if action_type == "apply_patch":
        return "patch_choice"
    if action_type == "verify":
        return "verification_planning"
    return "workflow_policy"


def normalize_vulnerability_label(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return VULNERABILITY_LABEL_ALIASES.get(normalized, normalized)


def build_instruction(
    *,
    task_id: str,
    instruction: str,
    code_snippet: str = "",
    scanner_hint: str = "",
    patch_options: Iterable[dict[str, str]] | None = None,
    episode_state: str | None = None,
    evidence: Iterable[str] | None = None,
) -> str:
    """Build a consistent prompt for SFT examples."""

    sections = [
        "You are an AI assistant helping to fix web security vulnerabilities.",
        f"Task: {task_id}",
        f"Instruction: {instruction}",
    ]

    if code_snippet:
        sections.append(f"Code to review:\n```\n{code_snippet}\n```")
    if scanner_hint:
        sections.append(f"Scanner hint: {scanner_hint}")
    if patch_options:
        sections.append(_format_patch_options(patch_options))
    if episode_state:
        sections.append(f"Episode state:\n{episode_state}")
    evidence_list = list(evidence or [])
    if evidence_list:
        sections.append("Evidence:\n" + "\n".join(f"- {item}" for item in evidence_list))

    sections.append(RESPONSE_SCHEMA_TEXT)
    return "\n\n".join(section for section in sections if section)


def _clean_review(review: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(review, dict):
        return {}

    cleaned: dict[str, Any] = {}
    for key in ("owasp_top10_2021", "cwe_id", "fix_family", "rationale"):
        value = review.get(key)
        if isinstance(value, str) and value.strip():
            cleaned[key] = value.strip()

    verification_plan = review.get("verification_plan")
    if isinstance(verification_plan, list):
        plan_items = [
            str(item).strip()
            for item in verification_plan
            if str(item).strip()
        ]
        if plan_items:
            cleaned["verification_plan"] = plan_items

    cvss = review.get("cvss_v3_1")
    if isinstance(cvss, dict):
        vector = cvss.get("vector")
        base_score = cvss.get("base_score")
        if isinstance(vector, str) and vector.strip() and isinstance(base_score, (int, float)):
            cleaned["cvss_v3_1"] = {
                "base_score": float(base_score),
                "vector": vector.strip(),
            }

    return cleaned


def build_dual_response(
    *,
    action_type: str,
    vulnerability_type: str | None = None,
    patch_id: str | None = None,
    review: dict[str, Any] | None = None,
) -> str:
    """Build the canonical dual-output training response."""

    if action_type not in ACTION_TYPES:
        raise ValueError(f"Unsupported action_type: {action_type}")

    action: dict[str, Any] = {"action_type": action_type}
    normalized_vulnerability_type = normalize_vulnerability_label(vulnerability_type)
    if normalized_vulnerability_type:
        action["vulnerability_type"] = normalized_vulnerability_type
    if patch_id:
        action["patch_id"] = patch_id

    payload: dict[str, Any] = {"action": action}
    cleaned_review = _clean_review(review)
    if cleaned_review:
        payload["review"] = cleaned_review

    return json.dumps(payload, sort_keys=True)


def _coerce_payload(value: Any, *, field_name: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"{field_name} must be a JSON object or JSON-encoded object")


def _normalize_training_output(raw_output: Any) -> tuple[str, str, str | None]:
    payload = _coerce_payload(raw_output, field_name="output")
    action_source = payload.get("action") if isinstance(payload.get("action"), dict) else payload
    review_source = payload.get("review")

    action_type = str(action_source.get("action_type", "")).strip() or "inspect"
    if action_type not in ACTION_TYPES:
        raise ValueError(f"Unsupported action_type in training example: {action_type}")

    vulnerability_type = normalize_vulnerability_label(action_source.get("vulnerability_type"))

    patch_id = action_source.get("patch_id")
    if patch_id is not None:
        patch_id = str(patch_id).strip() or None

    response = build_dual_response(
        action_type=action_type,
        vulnerability_type=vulnerability_type,
        patch_id=patch_id,
        review=review_source,
    )
    return response, action_type, vulnerability_type


def _extract_action_metadata(response: str) -> tuple[str, str | None, str | None]:
    payload = _coerce_payload(response, field_name="response")
    action_source = payload.get("action") if isinstance(payload.get("action"), dict) else payload

    action_type = str(action_source.get("action_type", "")).strip() or "inspect"
    if action_type not in ACTION_TYPES:
        action_type = "inspect"

    vulnerability_type = normalize_vulnerability_label(action_source.get("vulnerability_type"))

    patch_id = action_source.get("patch_id")
    if patch_id is not None:
        patch_id = str(patch_id).strip() or None

    return action_type, vulnerability_type, patch_id


def load_jsonl_dataset(
    filepath: Path | str,
    *,
    source: str = JSONL_SOURCE,
    preserve_row_source: bool = False,
) -> Dataset:
    """Load the primary WebSec JSONL dataset into the internal SFT format."""

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JSONL dataset not found: {filepath}")

    examples: list[dict[str, Any]] = []
    with filepath.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            task_id = str(record.get("task_id") or f"jsonl_row_{line_number}").strip()
            instruction_text = str(record.get("instruction") or "Review the code and choose the next safe action.").strip()
            input_block = record.get("input")
            if input_block is None:
                input_block = {}
            input_block = _coerce_payload(input_block, field_name="input")
            response, action_type, _ = _normalize_training_output(record.get("output", {}))
            example_source = source
            if preserve_row_source and record.get("source"):
                example_source = str(record["source"]).strip()
            example_family = str(record.get("example_family", "")).strip() or _infer_example_family(action_type)

            examples.append(
                _make_example(
                    task_id=task_id,
                    instruction=build_instruction(
                        task_id=task_id,
                        instruction=instruction_text,
                        code_snippet=str(input_block.get("code_snippet", "")).strip(),
                        scanner_hint=str(input_block.get("scanner_hint", "")).strip(),
                        patch_options=input_block.get("patch_options"),
                        episode_state=input_block.get("episode_state"),
                        evidence=input_block.get("evidence"),
                    ),
                    response=response,
                    action_type=action_type,
                    source=example_source,
                    example_family=example_family,
                )
            )

    dataset = _make_dataset(examples)
    print(f"✅ Loaded {len(dataset)} examples from {filepath.name}")
    return dataset


def summarize_dataset(dataset: Dataset) -> dict[str, Any]:
    """Return a compact summary for reporting before training."""

    action_counts = Counter()
    vulnerability_counts = Counter()
    source_counts = Counter()
    example_family_counts = Counter()

    if len(dataset) == 0:
        return {
            "total_examples": 0,
            "action_type_counts": {},
            "vulnerability_type_counts": {},
            "source_breakdown": {},
            "example_family_counts": {},
        }

    for index in range(len(dataset)):
        action_type, vulnerability_type, _ = _extract_action_metadata(dataset[index]["response"])
        action_counts[action_type] += 1
        if vulnerability_type:
            vulnerability_counts[vulnerability_type] += 1
        source_counts[str(dataset[index].get("source", "unknown"))] += 1
        example_family_counts[str(dataset[index].get("example_family", "unknown"))] += 1

    return {
        "total_examples": len(dataset),
        "action_type_counts": dict(action_counts),
        "vulnerability_type_counts": dict(vulnerability_counts),
        "source_breakdown": dict(source_counts),
        "example_family_counts": dict(example_family_counts),
    }


def _make_example(
    *,
    task_id: str,
    instruction: str,
    response: str,
    action_type: str,
    source: str,
    example_family: str,
) -> dict[str, Any]:
    return {
        "instruction": instruction,
        "response": response,
        "task_id": task_id,
        "action_type": action_type,
        "source": source,
        "example_family": example_family,
    }


RESEARCH_SCENARIOS: list[dict[str, Any]] = [
    {
        "task_id": "sqli_login",
        "title": "Patch the vulnerable login query",
        "instruction": "Fix SQL injection in the login handler.",
        "code_snippet": (
            "def login(username, password):\n"
            "    query = (\n"
            "        \"SELECT id FROM users WHERE username = '\" + username + \"' \"\n"
            "        \"AND password = '\" + password + \"'\"\n"
            "    )\n"
            "    return db.execute(query).fetchone()\n"
        ),
        "scanner_hint": "Semgrep warning: SQL injection risk from dynamic query construction.",
        "patch_options": [
            {"id": "parameterized_query", "label": "Use bound parameters for the login query"},
            {"id": "strip_quotes", "label": "Strip quotes from the username and password"},
            {"id": "disable_login", "label": "Disable the login route entirely"},
        ],
        "vulnerability_type": "sql_injection",
        "owasp_top10_2021": "A03: Injection",
        "cwe_id": "CWE-89",
        "cvss_v3_1": {
            "base_score": 8.6,
            "vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:N",
        },
        "patch_id": "parameterized_query",
        "fix_family": "parameterized_queries",
        "verification_plan": [
            "Run a login SQL injection payload and confirm the exploit is blocked.",
            "Verify valid logins still succeed with the parameterized query.",
            "Re-run the Semgrep SQL injection rule on the login handler.",
        ],
        "rationale": "The query concatenates attacker-controlled credentials directly into SQL.",
        "keywords": ["sql injection", "sqli", "semgrep", "cwe-89", "injection"],
    },
    {
        "task_id": "sqli_search",
        "title": "Fix SQL injection in product search",
        "instruction": "Fix SQL injection in the search endpoint while keeping fuzzy search working.",
        "code_snippet": (
            "def search_products(term):\n"
            "    sql = \"SELECT * FROM products WHERE name LIKE '%\" + term + \"%'\"\n"
            "    return db.execute(sql).fetchall()\n"
        ),
        "scanner_hint": "Scanner warning: LIKE clause built from untrusted input.",
        "patch_options": [
            {"id": "bind_like_parameter", "label": "Use a bound parameter for the LIKE clause"},
            {"id": "block_wildcards", "label": "Reject wildcard characters from the search term"},
            {"id": "disable_search", "label": "Disable search until the query is rewritten"},
        ],
        "vulnerability_type": "sql_injection",
        "owasp_top10_2021": "A03: Injection",
        "cwe_id": "CWE-89",
        "cvss_v3_1": {
            "base_score": 8.1,
            "vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:N",
        },
        "patch_id": "bind_like_parameter",
        "fix_family": "parameterized_queries",
        "verification_plan": [
            "Probe the search route with SQL wildcard and quote payloads.",
            "Confirm normal fuzzy search terms still return expected products.",
            "Re-run Semgrep and any search endpoint regression tests.",
        ],
        "rationale": "The search term is interpolated into SQL, so attacker input changes the query logic.",
        "keywords": ["sql injection", "search", "cwe-89", "injection"],
    },
    {
        "task_id": "xss_comments",
        "title": "Stop script execution in comment rendering",
        "instruction": "Fix XSS in the comment renderer without hiding normal comments.",
        "code_snippet": (
            "def render_comment(comment):\n"
            "    return f\"<div class='comment'>{comment}</div>\"\n"
        ),
        "scanner_hint": "Scanner warning: comment text is inserted into HTML without escaping.",
        "patch_options": [
            {"id": "html_escape", "label": "Escape comment text before rendering"},
            {"id": "remove_script_substring", "label": "Remove the literal 'script' substring"},
            {"id": "disable_comment_rendering", "label": "Hide comments completely"},
        ],
        "vulnerability_type": "xss",
        "owasp_top10_2021": "A03: Injection",
        "cwe_id": "CWE-79",
        "cvss_v3_1": {
            "base_score": 6.1,
            "vector": "CVSS:3.1/AV:N/AC:L/PR:L/UI:R/S:C/C:L/I:L/A:N",
        },
        "patch_id": "html_escape",
        "fix_family": "output_encoding",
        "verification_plan": [
            "Render a script-tag payload and confirm it is displayed as text.",
            "Render a normal comment and confirm the content still appears.",
            "Re-run the XSS Semgrep rule and UI regression checks.",
        ],
        "rationale": "User-controlled comment text is rendered into HTML without output encoding.",
        "keywords": ["xss", "cross-site scripting", "cwe-79", "html escape"],
    },
    {
        "task_id": "xss_profile",
        "title": "Escape untrusted profile markup",
        "instruction": "Fix XSS in the profile bio renderer while preserving formatting.",
        "code_snippet": (
            "def render_bio(user_bio):\n"
            "    return f\"<section class='bio'>{user_bio}</section>\"\n"
        ),
        "scanner_hint": "Scanner warning: profile bio content is reflected into HTML without escaping.",
        "patch_options": [
            {"id": "escape_profile_bio", "label": "Escape the profile bio before rendering"},
            {"id": "strip_angle_brackets", "label": "Strip angle brackets from the bio"},
            {"id": "remove_bio_feature", "label": "Disable profile bios"},
        ],
        "vulnerability_type": "xss",
        "owasp_top10_2021": "A03: Injection",
        "cwe_id": "CWE-79",
        "cvss_v3_1": {
            "base_score": 6.4,
            "vector": "CVSS:3.1/AV:N/AC:L/PR:L/UI:R/S:C/C:L/I:L/A:N",
        },
        "patch_id": "escape_profile_bio",
        "fix_family": "output_encoding",
        "verification_plan": [
            "Render a malicious profile bio payload and confirm script execution is blocked.",
            "Verify ordinary profile bios still render correctly after escaping.",
            "Re-run XSS scanning and profile integration tests.",
        ],
        "rationale": "The profile bio is treated as trusted HTML even though it is user supplied.",
        "keywords": ["xss", "profile", "cwe-79", "owasp"],
    },
    {
        "task_id": "broken_auth_admin",
        "title": "Restore admin-only access control",
        "instruction": "Fix missing role enforcement on the admin route.",
        "code_snippet": (
            "@app.get('/admin')\n"
            "def admin_panel(current_user):\n"
            "    if not current_user:\n"
            "        raise UnauthorizedError()\n"
            "    return render_admin_dashboard()\n"
        ),
        "scanner_hint": "Scanner warning: authenticated non-admin users can reach the admin route.",
        "patch_options": [
            {"id": "require_admin_role", "label": "Enforce an admin role check server-side"},
            {"id": "hide_admin_link", "label": "Hide the admin link in the client UI"},
            {"id": "deny_all_admin_access", "label": "Block every request to the admin route"},
        ],
        "vulnerability_type": "broken_auth",
        "owasp_top10_2021": "A01: Broken Access Control",
        "cwe_id": "CWE-862",
        "cvss_v3_1": {
            "base_score": 8.1,
            "vector": "CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:N",
        },
        "patch_id": "require_admin_role",
        "fix_family": "server_side_authorization",
        "verification_plan": [
            "Attempt admin access as a non-admin user and confirm access is denied.",
            "Verify legitimate admins still reach the dashboard.",
            "Re-run authorization tests and the relevant Semgrep access-control rules.",
        ],
        "rationale": "Authentication exists, but authorization for the admin role is missing.",
        "keywords": ["broken access control", "authorization", "admin", "cwe-862"],
    },
    {
        "task_id": "idor_order_lookup",
        "title": "Enforce resource ownership on order lookup",
        "instruction": "Fix the IDOR in the order lookup controller.",
        "code_snippet": (
            "def show(current_user, order_id):\n"
            "    raise Unauthorized unless current_user\n"
            "    Order.find(order_id)\n"
            "end\n"
        ),
        "scanner_hint": "Scanner warning: order lookup uses a direct object reference without an owner check.",
        "patch_options": [
            {"id": "enforce_resource_owner", "label": "Verify the current user owns the order"},
            {"id": "hide_identifier_client_side", "label": "Hide the order identifier in the client"},
            {"id": "disable_resource_route", "label": "Disable the order lookup route"},
        ],
        "vulnerability_type": "broken_access_control",
        "owasp_top10_2021": "A01: Broken Access Control",
        "cwe_id": "CWE-639",
        "cvss_v3_1": {
            "base_score": 8.0,
            "vector": "CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:L/A:N",
        },
        "patch_id": "enforce_resource_owner",
        "fix_family": "resource_authorization",
        "verification_plan": [
            "Request another user's order ID and confirm the lookup is blocked.",
            "Verify the rightful order owner still sees their own order.",
            "Re-run access-control checks and route tests.",
        ],
        "rationale": "The route uses a direct object reference without checking ownership.",
        "keywords": ["idor", "broken access control", "resource owner", "cwe-639"],
    },
    {
        "task_id": "path_traversal_download",
        "title": "Block path traversal in file download",
        "instruction": "Fix path traversal in the file download handler.",
        "code_snippet": (
            "def download(filename):\n"
            "    return open(BASE_DIR + '/' + filename, 'rb').read()\n"
        ),
        "scanner_hint": "Scanner warning: file paths are built from untrusted input without normalization.",
        "patch_options": [
            {"id": "normalize_and_bound_path", "label": "Normalize the path and enforce the base directory"},
            {"id": "strip_dotdot", "label": "Reject '..' substrings only"},
            {"id": "disable_downloads", "label": "Disable downloads entirely"},
        ],
        "vulnerability_type": "path_traversal",
        "owasp_top10_2021": "A01: Broken Access Control",
        "cwe_id": "CWE-22",
        "cvss_v3_1": {
            "base_score": 7.5,
            "vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N",
        },
        "patch_id": "normalize_and_bound_path",
        "fix_family": "path_canonicalization",
        "verification_plan": [
            "Attempt '../../' traversal payloads and confirm the handler rejects them.",
            "Verify legitimate files inside the allowed directory still download.",
            "Re-run path traversal checks and file-service tests.",
        ],
        "rationale": "Untrusted input controls the filesystem path without canonicalization or directory bounds.",
        "keywords": ["path traversal", "cwe-22", "base dir", "normalize"],
    },
    {
        "task_id": "ssrf_image_proxy",
        "title": "Constrain outbound fetches in the image proxy",
        "instruction": "Fix SSRF in the image proxy service.",
        "code_snippet": (
            "def proxy_image(url):\n"
            "    return requests.get(url, timeout=3).content\n"
        ),
        "scanner_hint": "Scanner warning: the service fetches attacker-controlled URLs directly.",
        "patch_options": [
            {"id": "allowlist_destinations", "label": "Allow only approved outbound hosts"},
            {"id": "strip_http_scheme", "label": "Remove the URL scheme from user input"},
            {"id": "disable_proxy_route", "label": "Disable the image proxy"},
        ],
        "vulnerability_type": "ssrf",
        "owasp_top10_2021": "A10: Server-Side Request Forgery",
        "cwe_id": "CWE-918",
        "cvss_v3_1": {
            "base_score": 8.8,
            "vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        },
        "patch_id": "allowlist_destinations",
        "fix_family": "outbound_request_allowlist",
        "verification_plan": [
            "Probe internal metadata and loopback URLs to confirm the proxy blocks them.",
            "Verify approved CDN image URLs still load successfully.",
            "Re-run SSRF tests and any outbound-request policy checks.",
        ],
        "rationale": "The proxy makes server-side requests to arbitrary attacker-controlled URLs.",
        "keywords": ["ssrf", "server-side request forgery", "owasp a10", "cwe-918"],
    },
    {
        "task_id": "command_injection_backup",
        "title": "Remove shell injection from backup job",
        "instruction": "Fix command injection in the backup task without breaking archive creation.",
        "code_snippet": (
            "def run_backup(archive_name, user_path):\n"
            "    subprocess.run(f\"tar -czf {archive_name} {user_path}\", shell=True)\n"
        ),
        "scanner_hint": "Scanner warning: user-controlled input reaches a shell command.",
        "patch_options": [
            {"id": "argument_list_no_shell", "label": "Pass an argument list without a shell"},
            {"id": "quote_user_input_only", "label": "Wrap the user path in extra quotes"},
            {"id": "disable_backups", "label": "Disable the backup feature"},
        ],
        "vulnerability_type": "command_injection",
        "owasp_top10_2021": "A03: Injection",
        "cwe_id": "CWE-78",
        "cvss_v3_1": {
            "base_score": 9.8,
            "vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        },
        "patch_id": "argument_list_no_shell",
        "fix_family": "shell_elimination",
        "verification_plan": [
            "Submit shell metacharacter payloads and confirm no extra commands execute.",
            "Verify legitimate backup archives are still created successfully.",
            "Re-run command-injection checks and task integration tests.",
        ],
        "rationale": "The backup job passes attacker-controlled values to a shell command.",
        "keywords": ["command injection", "shell", "cwe-78", "subprocess"],
    },
    {
        "task_id": "insecure_deserialization_session",
        "title": "Eliminate unsafe deserialization in session restore",
        "instruction": "Fix insecure deserialization in the session restore path.",
        "code_snippet": (
            "def restore_session(raw_cookie):\n"
            "    return pickle.loads(raw_cookie)\n"
        ),
        "scanner_hint": "Scanner warning: untrusted data is deserialized with pickle.",
        "patch_options": [
            {"id": "safe_json_session", "label": "Use JSON with schema validation instead of pickle"},
            {"id": "base64_wrap_pickle", "label": "Base64-encode the pickle payload first"},
            {"id": "disable_session_restore", "label": "Disable the restore endpoint"},
        ],
        "vulnerability_type": "insecure_deserialization",
        "owasp_top10_2021": "A08: Software and Data Integrity Failures",
        "cwe_id": "CWE-502",
        "cvss_v3_1": {
            "base_score": 8.8,
            "vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        },
        "patch_id": "safe_json_session",
        "fix_family": "safe_serialization",
        "verification_plan": [
            "Replay a malicious serialized payload and confirm arbitrary object loading is blocked.",
            "Verify valid sessions can still be restored from the new safe format.",
            "Re-run deserialization checks and session integration tests.",
        ],
        "rationale": "Untrusted serialized bytes are loaded directly into application objects.",
        "keywords": ["deserialization", "pickle", "cwe-502", "integrity failures"],
    },
    {
        "task_id": "template_injection_receipt",
        "title": "Stop server-side template injection in receipts",
        "instruction": "Fix template injection in the receipt preview renderer.",
        "code_snippet": (
            "def render_receipt(user_template, data):\n"
            "    return Template(user_template).render(**data)\n"
        ),
        "scanner_hint": "Scanner warning: user-provided template text is evaluated server-side.",
        "patch_options": [
            {"id": "use_fixed_template", "label": "Render receipts from a trusted fixed template"},
            {"id": "remove_double_braces", "label": "Strip template markers from the user input"},
            {"id": "disable_receipt_preview", "label": "Disable receipt preview"},
        ],
        "vulnerability_type": "template_injection",
        "owasp_top10_2021": "A03: Injection",
        "cwe_id": "CWE-1336",
        "cvss_v3_1": {
            "base_score": 8.8,
            "vector": "CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H",
        },
        "patch_id": "use_fixed_template",
        "fix_family": "trusted_template_only",
        "verification_plan": [
            "Attempt SSTI payloads and confirm template expressions are not executed.",
            "Verify legitimate receipt previews still render with approved placeholders.",
            "Re-run template-injection checks and rendering tests.",
        ],
        "rationale": "User-controlled template source is evaluated by the server-side template engine.",
        "keywords": ["template injection", "ssti", "cwe-1336", "template engine"],
    },
    {
        "task_id": "csrf_profile_update",
        "title": "Require CSRF protection on profile updates",
        "instruction": "Fix CSRF in the profile update endpoint.",
        "code_snippet": (
            "@app.post('/profile')\n"
            "def update_profile(request):\n"
            "    current_user.update(request.form)\n"
        ),
        "scanner_hint": "Scanner warning: profile updates do not validate a CSRF token.",
        "patch_options": [
            {"id": "require_csrf_token", "label": "Require a CSRF token on profile updates"},
            {"id": "check_referer_only", "label": "Validate only the Referer header"},
            {"id": "disable_profile_edits", "label": "Disable profile editing"},
        ],
        "vulnerability_type": "csrf",
        "owasp_top10_2021": "A01: Broken Access Control",
        "cwe_id": "CWE-352",
        "cvss_v3_1": {
            "base_score": 6.5,
            "vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:L/I:L/A:N",
        },
        "patch_id": "require_csrf_token",
        "fix_family": "anti_csrf_tokens",
        "verification_plan": [
            "Submit a forged cross-site profile update and confirm it is rejected.",
            "Verify normal same-site profile edits with valid tokens still succeed.",
            "Re-run web-security regression tests for profile updates.",
        ],
        "rationale": "A state-changing request can be forged because it lacks CSRF validation.",
        "keywords": ["csrf", "cwe-352", "token", "state-changing request"],
    },
    {
        "task_id": "hardcoded_secret_api",
        "title": "Remove hard-coded API credentials",
        "instruction": "Fix the hard-coded API key leak in the service client.",
        "code_snippet": (
            "API_KEY = 'sk_live_prod_123456'\n"
            "client = PaymentsClient(api_key=API_KEY)\n"
        ),
        "scanner_hint": "Trivy warning: hard-coded credential committed in source control.",
        "patch_options": [
            {"id": "move_secret_to_env", "label": "Load the secret from a managed environment variable"},
            {"id": "obfuscate_secret", "label": "Split the secret across multiple string literals"},
            {"id": "disable_payments", "label": "Disable the payments client entirely"},
        ],
        "vulnerability_type": "insecure_authentication",
        "owasp_top10_2021": "A02: Cryptographic Failures",
        "cwe_id": "CWE-798",
        "cvss_v3_1": {
            "base_score": 8.2,
            "vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:N",
        },
        "patch_id": "move_secret_to_env",
        "fix_family": "secret_externalization",
        "verification_plan": [
            "Confirm the secret is no longer present in source files or the built artifact.",
            "Verify the service still starts and reads the secret from the configured environment.",
            "Re-run Trivy secret scanning after rotating the leaked credential.",
        ],
        "rationale": "A production credential is committed directly in source code and can be exfiltrated.",
        "keywords": ["secret", "hard-coded", "trivy", "cwe-798", "credential"],
    },
    {
        "task_id": "vulnerable_dependency_frontend",
        "title": "Upgrade a vulnerable frontend dependency",
        "instruction": "Fix the vulnerable dependency without breaking the frontend build.",
        "code_snippet": (
            "{\n"
            "  \"dependencies\": {\n"
            "    \"express\": \"4.16.0\",\n"
            "    \"serialize-javascript\": \"2.1.1\"\n"
            "  }\n"
            "}\n"
        ),
        "scanner_hint": "Trivy warning: package lock contains a vulnerable dependency with a known advisory.",
        "patch_options": [
            {"id": "upgrade_dependency", "label": "Upgrade to a fixed dependency version"},
            {"id": "pin_transitive_only", "label": "Pin only one transitive package without validation"},
            {"id": "remove_feature", "label": "Delete the vulnerable feature completely"},
        ],
        "vulnerability_type": "vulnerable_dependency",
        "owasp_top10_2021": "A06: Vulnerable and Outdated Components",
        "cwe_id": "CWE-1104",
        "cvss_v3_1": {
            "base_score": 7.5,
            "vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:N",
        },
        "patch_id": "upgrade_dependency",
        "fix_family": "dependency_upgrade",
        "verification_plan": [
            "Re-run Trivy or SCA scanning and confirm the advisory no longer appears.",
            "Verify the frontend build and smoke tests still pass after the upgrade.",
            "Document the upgraded version and lockfile change.",
        ],
        "rationale": "The dependency manifest pins a component version with a known published vulnerability.",
        "keywords": ["dependency", "trivy", "owasp a06", "component", "advisory"],
    },
    {
        "task_id": "weak_password_storage",
        "title": "Replace weak password hashing",
        "instruction": "Fix weak password storage without breaking login verification.",
        "code_snippet": (
            "def store_password(password):\n"
            "    return hashlib.md5(password.encode()).hexdigest()\n"
        ),
        "scanner_hint": "Scanner warning: passwords are hashed with MD5 without a work factor.",
        "patch_options": [
            {"id": "argon2id_hash", "label": "Use Argon2id for password hashing"},
            {"id": "salt_md5_only", "label": "Add a static salt but keep MD5"},
            {"id": "disable_password_login", "label": "Disable password authentication"},
        ],
        "vulnerability_type": "insecure_authentication",
        "owasp_top10_2021": "A02: Cryptographic Failures",
        "cwe_id": "CWE-916",
        "cvss_v3_1": {
            "base_score": 7.4,
            "vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N",
        },
        "patch_id": "argon2id_hash",
        "fix_family": "memory_hard_hashing",
        "verification_plan": [
            "Verify stored hashes are generated with Argon2id and a per-user salt.",
            "Confirm existing password verification still works after migration or fallback logic.",
            "Re-run authentication tests and password-storage checks.",
        ],
        "rationale": "MD5 is too fast and unsuitable for password storage because it lacks a work factor.",
        "keywords": ["password", "md5", "cryptographic failures", "cwe-916"],
    },
    {
        "task_id": "xpath_injection_search",
        "title": "Bind untrusted values in XPath search",
        "instruction": "Fix XPath injection in the directory lookup endpoint.",
        "code_snippet": (
            "def find_user(username):\n"
            "    xpath = f\"//user[name/text()='{username}']\"\n"
            "    return xml_doc.xpath(xpath)\n"
        ),
        "scanner_hint": "Scanner warning: untrusted input is embedded into an XPath expression.",
        "patch_options": [
            {"id": "xpath_parameterization", "label": "Use a safe variable binding for the XPath query"},
            {"id": "escape_quotes_only", "label": "Escape only quote characters in the username"},
            {"id": "disable_directory_lookup", "label": "Disable the lookup endpoint"},
        ],
        "vulnerability_type": "xpath_injection",
        "owasp_top10_2021": "A03: Injection",
        "cwe_id": "CWE-643",
        "cvss_v3_1": {
            "base_score": 8.1,
            "vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:N",
        },
        "patch_id": "xpath_parameterization",
        "fix_family": "query_parameterization",
        "verification_plan": [
            "Probe the XPath expression with quote and predicate payloads.",
            "Verify valid user lookups still work with safe variable binding.",
            "Re-run injection-focused scanning for the directory endpoint.",
        ],
        "rationale": "User input becomes part of the XPath expression and can change the query semantics.",
        "keywords": ["xpath", "injection", "cwe-643", "query parameterization"],
    },
    {
        "task_id": "dangerous_file_upload",
        "title": "Restrict dangerous file uploads",
        "instruction": "Fix unrestricted file upload without breaking safe image uploads.",
        "code_snippet": (
            "def upload(file):\n"
            "    path = os.path.join('static/uploads', file.filename)\n"
            "    file.save(path)\n"
        ),
        "scanner_hint": "Scanner warning: uploads are stored under a web-served path without file-type validation.",
        "patch_options": [
            {"id": "validate_type_store_outside_webroot", "label": "Validate type and store uploads outside the web root"},
            {"id": "rename_extension_only", "label": "Rename the file extension but keep the same storage path"},
            {"id": "disable_uploads", "label": "Disable uploads entirely"},
        ],
        "vulnerability_type": "dangerous_file_upload",
        "owasp_top10_2021": "A05: Security Misconfiguration",
        "cwe_id": "CWE-434",
        "cvss_v3_1": {
            "base_score": 8.8,
            "vector": "CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H",
        },
        "patch_id": "validate_type_store_outside_webroot",
        "fix_family": "upload_type_and_storage_validation",
        "verification_plan": [
            "Upload a dangerous script file and confirm it cannot be served or executed.",
            "Verify allowed image uploads still succeed and remain retrievable.",
            "Re-run upload validation tests and any relevant security scan rules.",
        ],
        "rationale": "The upload path is web accessible and the code does not validate type or storage location.",
        "keywords": ["file upload", "webroot", "cwe-434", "misconfiguration"],
    },
]

SCENARIO_BY_TASK_ID = {scenario["task_id"]: scenario for scenario in RESEARCH_SCENARIOS}
CANONICAL_ENV_TASKS = ("sqli_login", "xss_comments", "broken_auth_admin")


def _load_research_document_text(filepath: Path | str) -> str:
    path = Path(filepath)
    if path.suffix.lower() not in {".md", ".txt"}:
        raise ValueError(f"Research document must be .md or .txt, got: {path}")
    return path.read_text(encoding="utf-8")


def _research_doc_profile(text: str) -> dict[str, bool]:
    lowered = text.lower()
    return {
        "mentions_semgrep": "semgrep" in lowered,
        "mentions_trivy": "trivy" in lowered,
        "mentions_zap": "owasp zap" in lowered or "zap" in lowered,
        "mentions_cvss": "cvss" in lowered,
        "mentions_cwe": "cwe" in lowered,
        "mentions_owasp": "owasp" in lowered,
        "mentions_closed_loop": all(
            token in lowered for token in ("inspect", "classify", "patch", "verify", "submit")
        ),
    }


def _select_report_scenarios(text: str) -> list[dict[str, Any]]:
    lowered = text.lower()
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for index, scenario in enumerate(RESEARCH_SCENARIOS):
        score = sum(1 for keyword in scenario["keywords"] if keyword in lowered)
        ranked.append((score, index, scenario))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for score, _, scenario in ranked:
        if score <= 0:
            continue
        selected.append(scenario)
        seen_ids.add(scenario["task_id"])

    for scenario in RESEARCH_SCENARIOS:
        if scenario["task_id"] not in seen_ids:
            selected.append(scenario)
            seen_ids.add(scenario["task_id"])

    return selected[:15]


def _verification_plan_for_scenario(
    scenario: dict[str, Any],
    profile: dict[str, bool],
) -> list[str]:
    plan = list(scenario["verification_plan"])
    lowered_title = scenario["title"].lower()

    if profile["mentions_trivy"] and any(
        token in lowered_title or token in scenario["task_id"]
        for token in ("secret", "dependency", "upload")
    ):
        plan.append("Re-run Trivy and confirm the relevant secret/dependency finding is gone.")
    if profile["mentions_semgrep"] and "Semgrep" not in " ".join(plan):
        plan.append("Re-run the Semgrep rule associated with the original finding.")
    if profile["mentions_zap"] and any(
        token in scenario["task_id"] for token in ("xss", "csrf", "ssrf")
    ):
        plan.append("Run the ZAP automation regression plan against the affected route.")

    deduped: list[str] = []
    for item in plan:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _taxonomy_example(
    scenario: dict[str, Any],
    profile: dict[str, bool],
) -> dict[str, Any]:
    instruction = build_instruction(
        task_id=scenario["task_id"],
        instruction=(
            "Classify the vulnerability, map it to the correct OWASP/CWE labels, "
            "and keep the explanation short."
        ),
        code_snippet=scenario["code_snippet"],
        scanner_hint=scenario["scanner_hint"],
        evidence=[
            "Closed-loop policy: the patch is only valid after exploit blocking and functionality checks.",
            "Use the research playbook's taxonomy guidance for OWASP, CWE, and CVSS.",
        ],
    )
    response = build_dual_response(
        action_type="classify",
        vulnerability_type=scenario["vulnerability_type"],
        review={
            "owasp_top10_2021": scenario["owasp_top10_2021"] if profile["mentions_owasp"] else None,
            "cwe_id": scenario["cwe_id"] if profile["mentions_cwe"] else None,
            "cvss_v3_1": scenario["cvss_v3_1"] if profile["mentions_cvss"] else None,
            "rationale": scenario["rationale"],
        },
    )
    return _make_example(
        task_id=scenario["task_id"],
        instruction=instruction,
        response=response,
        action_type="classify",
        source="research_doc",
        example_family="taxonomy_mapping",
    )


def _patch_choice_example(
    scenario: dict[str, Any],
    profile: dict[str, bool],
) -> dict[str, Any]:
    instruction = build_instruction(
        task_id=scenario["task_id"],
        instruction=(
            "Choose the safest patch that blocks the exploit and preserves expected behavior."
        ),
        code_snippet=scenario["code_snippet"],
        scanner_hint=scenario["scanner_hint"],
        patch_options=scenario["patch_options"],
        evidence=[
            "Weak filtering or UI-only fixes are insufficient.",
            "Do not choose fixes that remove the feature if a safe repair exists.",
        ],
    )
    response = build_dual_response(
        action_type="apply_patch",
        vulnerability_type=scenario["vulnerability_type"],
        patch_id=scenario["patch_id"],
        review={
            "owasp_top10_2021": scenario["owasp_top10_2021"] if profile["mentions_owasp"] else None,
            "cwe_id": scenario["cwe_id"] if profile["mentions_cwe"] else None,
            "fix_family": scenario["fix_family"],
            "rationale": scenario["rationale"],
        },
    )
    return _make_example(
        task_id=scenario["task_id"],
        instruction=instruction,
        response=response,
        action_type="apply_patch",
        source="research_doc",
        example_family="patch_choice",
    )


def _verification_example(
    scenario: dict[str, Any],
    profile: dict[str, bool],
) -> dict[str, Any]:
    instruction = build_instruction(
        task_id=scenario["task_id"],
        instruction=(
            f"The patch '{scenario['patch_id']}' has been selected. Plan the verification step."
        ),
        code_snippet=scenario["code_snippet"],
        scanner_hint=scenario["scanner_hint"],
        patch_options=scenario["patch_options"],
        evidence=[
            f"Chosen fix family: {scenario['fix_family']}",
            "Verification must prove the exploit is blocked and normal behavior still works.",
        ],
    )
    response = build_dual_response(
        action_type="verify",
        patch_id=scenario["patch_id"],
        review={
            "owasp_top10_2021": scenario["owasp_top10_2021"] if profile["mentions_owasp"] else None,
            "cwe_id": scenario["cwe_id"] if profile["mentions_cwe"] else None,
            "fix_family": scenario["fix_family"],
            "verification_plan": _verification_plan_for_scenario(scenario, profile),
            "rationale": "Verification closes the loop by testing exploit blocking, functionality, and rescan hygiene.",
        },
    )
    return _make_example(
        task_id=scenario["task_id"],
        instruction=instruction,
        response=response,
        action_type="verify",
        source="research_doc",
        example_family="verification_planning",
    )


def _workflow_example(
    scenario: dict[str, Any],
    profile: dict[str, bool],
    index: int,
) -> dict[str, Any]:
    workflow_variants = [
        {
            "episode_state": "New episode. The task exists, but the code and scanner hint have not been revealed yet.",
            "action_type": "inspect",
            "review": {
                "rationale": "Inspect first so the agent can gather the vulnerable code, scanner hint, and patch options.",
            },
        },
        {
            "episode_state": "The code and scanner hint are visible, but no vulnerability label has been chosen yet.",
            "action_type": "classify",
            "review": {
                "owasp_top10_2021": scenario["owasp_top10_2021"] if profile["mentions_owasp"] else None,
                "cwe_id": scenario["cwe_id"] if profile["mentions_cwe"] else None,
                "cvss_v3_1": scenario["cvss_v3_1"] if profile["mentions_cvss"] else None,
                "rationale": "Choose the precise vulnerability label before selecting a patch.",
            },
        },
        {
            "episode_state": "The vulnerability has been classified correctly, but no patch has been applied.",
            "action_type": "apply_patch",
            "review": {
                "fix_family": scenario["fix_family"],
                "rationale": "After correct classification, the next step is to apply the safest patch option.",
            },
        },
        {
            "episode_state": "A patch has been applied, but exploit blocking and functionality checks have not run yet.",
            "action_type": "verify",
            "review": {
                "fix_family": scenario["fix_family"],
                "verification_plan": _verification_plan_for_scenario(scenario, profile),
                "rationale": "The repair is not trusted until verification proves it is safe and effective.",
            },
        },
        {
            "episode_state": "Classification is correct, the safe patch is applied, and verification already passed.",
            "action_type": "submit",
            "review": {
                "rationale": "All required checks have passed, so the agent should submit the final solution.",
            },
        },
    ]

    variant = workflow_variants[index % len(workflow_variants)]
    action_type = variant["action_type"]
    response = build_dual_response(
        action_type=action_type,
        vulnerability_type=scenario["vulnerability_type"] if action_type == "classify" else None,
        patch_id=scenario["patch_id"] if action_type in {"apply_patch", "verify"} else None,
        review=variant["review"],
    )
    instruction = build_instruction(
        task_id=scenario["task_id"],
        instruction="Choose the next action in the closed-loop WebSec repair workflow.",
        code_snippet=scenario["code_snippet"],
        scanner_hint=scenario["scanner_hint"],
        patch_options=scenario["patch_options"] if action_type == "apply_patch" else None,
        episode_state=variant["episode_state"],
        evidence=["Follow the inspect → classify → apply_patch → verify → submit loop."],
    )
    return _make_example(
        task_id=scenario["task_id"],
        instruction=instruction,
        response=response,
        action_type=action_type,
        source="research_doc",
        example_family="workflow_policy",
    )


def create_report_training_data(filepath: Path | str) -> Dataset:
    """Convert a markdown/txt security research writeup into SFT examples."""

    text = _load_research_document_text(filepath)
    profile = _research_doc_profile(text)
    scenarios = _select_report_scenarios(text)

    examples: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        examples.append(_taxonomy_example(scenario, profile))
        examples.append(_patch_choice_example(scenario, profile))
        examples.append(_verification_example(scenario, profile))
        examples.append(_workflow_example(scenario, profile, index))

    dataset = _make_dataset(examples)
    print(f"✅ Built {len(dataset)} report-derived examples from {Path(filepath).name}")
    return dataset


def load_trajectory_metrics(metrics_file: Path) -> list[dict]:
    """Load episode metrics from baseline or trained runs."""

    with open(metrics_file, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("episodes", [])


def create_training_dataset(
    metrics_files: list[Path],
    task_definitions_file: Path | None = None,
) -> Dataset:
    """Create a supervised dataset from coarse episode metrics."""

    training_examples: list[dict[str, Any]] = []
    task_info: dict[str, Any] = {}
    if task_definitions_file and task_definitions_file.exists():
        with open(task_definitions_file, encoding="utf-8") as f:
            task_data = json.load(f)
        task_info = {task["id"]: task for task in task_data.get("tasks", [])}

    for metrics_file in metrics_files:
        episodes = load_trajectory_metrics(metrics_file)
        for episode in episodes:
            if not episode.get("success", False):
                continue

            task_id = episode.get("task_id", "")
            scenario = SCENARIO_BY_TASK_ID.get(task_id, {})
            task_def = task_info.get(task_id, {})
            action_type = "classify" if episode.get("vulnerability_correct") else "inspect"

            instruction = build_instruction(
                task_id=task_id,
                instruction=task_def.get("instruction", f"Choose the next safe action for {task_id}."),
                code_snippet=task_def.get("code_snippet", scenario.get("code_snippet", "# Code unavailable")),
                scanner_hint=task_def.get("scanner_hint", scenario.get("scanner_hint", "")),
                evidence=[
                    f"Episode outcome: success={episode.get('success', False)}",
                    f"Patch correct: {episode.get('patch_correct', False)}",
                ],
            )
            response = build_dual_response(
                action_type=action_type,
                vulnerability_type=scenario.get("vulnerability_type") if action_type == "classify" else None,
                review={
                    "owasp_top10_2021": scenario.get("owasp_top10_2021"),
                    "cwe_id": scenario.get("cwe_id"),
                    "cvss_v3_1": scenario.get("cvss_v3_1"),
                    "rationale": "Successful episodes teach the next safe step to take in the repair loop.",
                },
            )
            training_examples.append(
                _make_example(
                    task_id=task_id,
                    instruction=instruction,
                    response=response,
                    action_type=action_type,
                    source="metrics",
                    example_family="workflow_policy",
                )
            )

    return _make_dataset(training_examples)


def create_synthetic_training_data() -> Dataset:
    """Create canonical synthetic training data for the three deterministic env tasks."""

    examples: list[dict[str, Any]] = []
    for task_id in CANONICAL_ENV_TASKS:
        scenario = SCENARIO_BY_TASK_ID[task_id]
        examples.append(
            _make_example(
                task_id=task_id,
                instruction=build_instruction(
                    task_id=task_id,
                    instruction=scenario["instruction"],
                    code_snippet=scenario["code_snippet"],
                    scanner_hint=scenario["scanner_hint"],
                ),
                response=build_dual_response(
                    action_type="classify",
                    vulnerability_type=scenario["vulnerability_type"],
                    review={
                        "owasp_top10_2021": scenario["owasp_top10_2021"],
                        "cwe_id": scenario["cwe_id"],
                        "cvss_v3_1": scenario["cvss_v3_1"],
                        "rationale": scenario["rationale"],
                    },
                ),
                action_type="classify",
                source="synthetic",
                example_family="taxonomy_mapping",
            )
        )
        examples.append(
            _make_example(
                task_id=task_id,
                instruction=build_instruction(
                    task_id=task_id,
                    instruction="Pick the safest patch for the current vulnerability.",
                    code_snippet=scenario["code_snippet"],
                    scanner_hint=scenario["scanner_hint"],
                    patch_options=scenario["patch_options"],
                ),
                response=build_dual_response(
                    action_type="apply_patch",
                    vulnerability_type=scenario["vulnerability_type"],
                    patch_id=scenario["patch_id"],
                    review={
                        "fix_family": scenario["fix_family"],
                        "rationale": scenario["rationale"],
                    },
                ),
                action_type="apply_patch",
                source="synthetic",
                example_family="patch_choice",
            )
        )

    return _make_dataset(examples)


def load_trajectory_logs(filepath: Path | str) -> Dataset:
    """Load full environment trajectory logs into dual-output SFT examples."""

    filepath = Path(filepath)
    if not filepath.exists():
        print(f"⚠️  Trajectory logs not found at {filepath}. Returning empty dataset.")
        return _empty_dataset()

    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    trajectories = data.get("trajectories", [])
    examples: list[dict[str, Any]] = []

    for trajectory in trajectories:
        task_id = trajectory.get("task_id", "unknown")
        scenario = SCENARIO_BY_TASK_ID.get(task_id, {})
        vuln_type = normalize_vulnerability_label(
            trajectory.get("vulnerability_type") or scenario.get("vulnerability_type")
        )
        steps = trajectory.get("steps", [])

        for step_index, step in enumerate(steps, start=1):
            action_type = step.get("action_type", "inspect")
            if action_type not in ACTION_TYPES:
                continue

            review: dict[str, Any] = {
                "rationale": str(step.get("observation", "")).strip() or "Use the trajectory signal to choose the next safe action.",
            }
            patch_id = None
            if action_type == "classify":
                review.update(
                    {
                        "owasp_top10_2021": scenario.get("owasp_top10_2021"),
                        "cwe_id": scenario.get("cwe_id"),
                        "cvss_v3_1": scenario.get("cvss_v3_1"),
                    }
                )
            elif action_type == "apply_patch":
                patch_id = scenario.get("patch_id")
                review["fix_family"] = scenario.get("fix_family")
            elif action_type == "verify":
                patch_id = scenario.get("patch_id")
                review["fix_family"] = scenario.get("fix_family")
                review["verification_plan"] = scenario.get("verification_plan")

            instruction = build_instruction(
                task_id=task_id,
                instruction="Use the current trajectory observation to decide the next WebSec repair action.",
                code_snippet=scenario.get("code_snippet", ""),
                scanner_hint=scenario.get("scanner_hint", ""),
                patch_options=scenario.get("patch_options") if action_type == "apply_patch" else None,
                episode_state=(
                    f"Trajectory step: {step_index}\n"
                    f"Observed reward: {step.get('reward', 0.0)}\n"
                    f"Completed: {step.get('done', False)}"
                ),
                evidence=[str(step.get("observation", ""))],
            )
            response = build_dual_response(
                action_type=action_type,
                vulnerability_type=vuln_type if action_type == "classify" else None,
                patch_id=patch_id,
                review=review,
            )
            examples.append(
                _make_example(
                    task_id=task_id,
                    instruction=instruction,
                    response=response,
                    action_type=action_type,
                    source="trajectory",
                    example_family="workflow_policy",
                )
            )

    if not examples:
        print(f"⚠️  No examples found in {filepath}. Returning empty dataset.")
        return _empty_dataset()

    print(f"✅ Loaded {len(examples)} examples from {len(trajectories)} trajectories")
    return _make_dataset(examples)


def load_swe_bench_lite() -> Dataset:
    """Load SWE-bench Lite as a generic code-fix dataset."""

    try:
        print("📥 Downloading SWE-bench Lite from Hugging Face...")
        from datasets import load_dataset

        raw_dataset = load_dataset("princeton-nlp/SWE-bench", "lite", split="test")
        examples: list[dict[str, Any]] = []

        for item in raw_dataset:
            problem_statement = item.get("problem_statement", "")[:500]
            instance_id = item.get("instance_id", "unknown")
            instruction = build_instruction(
                task_id=instance_id,
                instruction="Choose the next repair action for this software issue.",
                scanner_hint=problem_statement,
                evidence=[
                    "This example comes from a real GitHub bug-fix benchmark.",
                    "Prioritize the action that moves the issue toward a verified patch.",
                ],
            )
            response = build_dual_response(
                action_type="apply_patch",
                review={
                    "fix_family": "validated_code_change",
                    "rationale": "Real bug-fix benchmarks teach the model to move from diagnosis toward a concrete repair.",
                },
            )
            examples.append(
                _make_example(
                    task_id=instance_id,
                    instruction=instruction,
                    response=response,
                    action_type="apply_patch",
                    source="swe_bench_lite",
                    example_family="patch_choice",
                )
            )

        if not examples:
            print("⚠️  No examples loaded from SWE-bench Lite. Returning empty dataset.")
            return _empty_dataset()

        print(f"✅ Loaded {len(examples)} examples from SWE-bench Lite")
        return _make_dataset(examples)

    except Exception as exc:
        print(f"⚠️  Could not load SWE-bench Lite: {exc}")
        print("   Continuing without SWE-bench data...")
        return _empty_dataset()


def _repeat_to_size(dataset: Dataset, target_size: int) -> Dataset:
    if len(dataset) == 0 or target_size <= 0:
        return _empty_dataset()
    if len(dataset) >= target_size:
        return dataset.select(range(target_size))

    repeats, remainder = divmod(target_size, len(dataset))
    indices = list(range(len(dataset))) * repeats + list(range(remainder))
    return dataset.select(indices)


def _cap_dataset(dataset: Dataset, max_size: int) -> Dataset:
    if max_size <= 0 or len(dataset) == 0:
        return _empty_dataset()
    if len(dataset) <= max_size:
        return dataset
    return dataset.select(range(max_size))


def _select_dataset_by_sources(dataset: Dataset, source_names: set[str]) -> Dataset:
    if len(dataset) == 0:
        return _empty_dataset()
    indices = [
        index
        for index, source in enumerate(dataset["source"])
        if str(source) in source_names
    ]
    if not indices:
        return _empty_dataset()
    return dataset.select(indices)


def _rebalance_external_dataset(external_dataset: Dataset, custom_dataset_size: int) -> Dataset:
    if len(external_dataset) == 0:
        return _empty_dataset()

    if custom_dataset_size <= 0:
        return external_dataset

    desired_total = max(custom_dataset_size, math.ceil(custom_dataset_size / SOURCE_GROUP_TARGET_SHARES["custom"]))
    rebalanced_groups: list[Dataset] = []
    custom_like_dataset = _select_dataset_by_sources(external_dataset, SOURCE_GROUPS["custom"])
    if len(custom_like_dataset) > 0:
        rebalanced_groups.append(custom_like_dataset)

    for group_name in ("primevul_devign", "juliet_owasp", "patch_pairs"):
        group_dataset = _select_dataset_by_sources(external_dataset, SOURCE_GROUPS[group_name])
        target_size = math.ceil(desired_total * SOURCE_GROUP_TARGET_SHARES[group_name])
        if len(group_dataset) == 0:
            continue
        rebalanced_groups.append(_cap_dataset(group_dataset, target_size))

    if not rebalanced_groups:
        return _empty_dataset()

    return _concat_datasets(rebalanced_groups)


def create_combined_dataset(
    jsonl_dataset_path: Path | str | None = None,
    trajectory_logs_path: Path | str | None = None,
    research_doc_path: Path | str | None = None,
    external_websec_dataset_path: Path | str | None = None,
    include_swe_bench: bool = True,
    include_synthetic: bool = True,
) -> Dataset:
    """Create a balanced mixed dataset for Colab training.

    Guarantees:
    - report-derived examples are at least 30% of the final dataset
    - SWE-bench is capped at 25% of the final dataset
    """

    jsonl_dataset = load_jsonl_dataset(jsonl_dataset_path) if jsonl_dataset_path else _empty_dataset()
    trajectory_dataset = load_trajectory_logs(trajectory_logs_path) if trajectory_logs_path else _empty_dataset()
    research_dataset = create_report_training_data(research_doc_path) if research_doc_path else _empty_dataset()
    synthetic_dataset = create_synthetic_training_data() if include_synthetic else _empty_dataset()
    external_dataset = (
        load_jsonl_dataset(
            external_websec_dataset_path,
            source="external_websec",
            preserve_row_source=True,
        )
        if external_websec_dataset_path
        else _empty_dataset()
    )
    swe_dataset = load_swe_bench_lite() if include_swe_bench else _empty_dataset()

    base_non_report = _concat_datasets(
        [
            dataset
            for dataset in (jsonl_dataset, trajectory_dataset, synthetic_dataset)
            if len(dataset) > 0
        ]
    )

    target_report_count = len(research_dataset)
    if len(research_dataset) > 0:
        minimum_report_count = math.ceil((2 * len(base_non_report)) / 3)
        target_report_count = max(len(research_dataset), minimum_report_count)
        research_dataset = _repeat_to_size(research_dataset, target_report_count)

    external_dataset = _rebalance_external_dataset(external_dataset, len(base_non_report) + len(research_dataset))

    non_swe_datasets = [
        dataset for dataset in (base_non_report, research_dataset, external_dataset)
        if len(dataset) > 0
    ]
    non_swe_dataset = _concat_datasets(non_swe_datasets)

    swe_cap = len(non_swe_dataset) // 3 if len(non_swe_dataset) > 0 else 0
    swe_dataset = _cap_dataset(swe_dataset, swe_cap)

    datasets_to_combine = [
        dataset
        for dataset in (non_swe_dataset, swe_dataset)
        if len(dataset) > 0
    ]

    if not datasets_to_combine:
        print("⚠️  No datasets available. Returning synthetic data only.")
        return _empty_dataset()

    combined = _concat_datasets(datasets_to_combine)
    report_count = sum(1 for source in combined["source"] if source == "research_doc")
    swe_count = sum(1 for source in combined["source"] if source == "swe_bench_lite")
    total_count = len(combined)

    print(f"✅ Added {len(jsonl_dataset)} examples from JSONL dataset")
    print(f"✅ Added {len(trajectory_dataset)} examples from trajectory logs")
    print(f"✅ Added {len(synthetic_dataset)} synthetic examples")
    if research_doc_path:
        print(f"✅ Added {report_count} report-derived examples")
    if external_websec_dataset_path:
        print(f"✅ Added {len(external_dataset)} external WebSec examples")
    if include_swe_bench:
        print(f"✅ Added {swe_count} SWE-bench Lite examples")
    print(f"\n📊 Total training examples: {total_count}")

    if total_count > 0:
        print(f"   Report share: {report_count / total_count:.1%}")
        print(f"   SWE-bench share: {swe_count / total_count:.1%}")

    return combined
