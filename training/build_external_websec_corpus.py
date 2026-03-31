#!/usr/bin/env python3
"""Build a normalized external WebSec corpus for QLoRA training.

This script converts local copies or exports of defensive WebSec datasets into
the project JSONL schema:

{
  "task_id": "...",
  "instruction": "...",
  "input": {
    "code_snippet": "...",
    "scanner_hint": "...",
    "patch_options": [...],
    "evidence": [...]
  },
  "output": {
    "action": {...},
    "review": {...}
  },
  "source": "...",
  "example_family": "..."
}

Supported source families:
- PrimeVul / Devign style vulnerability records
- Juliet / SARD or OWASP Benchmark style directories or exports
- Big-Vul / CVEfixes / GitHub security fix style patch pairs
- curated NVD / ExploitDB exports
- built-in synthetic web patterns for coverage
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

RELEVANT_VULNERABILITY_TYPES = {
    "sql_injection",
    "xss",
    "broken_auth",
    "insecure_authentication",
    "path_traversal",
    "file_inclusion",
    "command_injection",
    "broken_access_control",
    "insecure_deserialization",
    "csrf",
}

LABEL_ALIASES = {
    "auth_bypass": "broken_auth",
    "authentication_bypass": "broken_auth",
    "access_control": "broken_access_control",
    "idor": "broken_access_control",
    "directory_traversal": "path_traversal",
    "command_execution": "command_injection",
    "os_command_injection": "command_injection",
    "hardcoded_secret": "insecure_authentication",
    "hardcoded_credentials": "insecure_authentication",
    "credentials": "insecure_authentication",
    "rfi": "file_inclusion",
    "lfi": "file_inclusion",
    "deserialization": "insecure_deserialization",
}

CWE_TO_LABEL = {
    "22": "path_traversal",
    "78": "command_injection",
    "79": "xss",
    "89": "sql_injection",
    "98": "file_inclusion",
    "287": "broken_auth",
    "352": "csrf",
    "502": "insecure_deserialization",
    "639": "broken_access_control",
    "798": "insecure_authentication",
    "862": "broken_access_control",
}

VULNERABILITY_METADATA = {
    "sql_injection": {
        "owasp_top10_2021": "A03: Injection",
        "cwe_id": "CWE-89",
        "patch_id": "parameterized_query",
        "fix_family": "parameterized_queries",
        "patch_options": [
            {"id": "parameterized_query", "label": "Use bound parameters for the database query"},
            {"id": "escape_quotes_only", "label": "Escape only quote characters in the user input"},
            {"id": "disable_endpoint", "label": "Disable the route instead of fixing the query"},
        ],
        "verification_plan": [
            "Replay a SQL injection payload and confirm the query logic cannot be altered.",
            "Verify the valid request path still works with the parameterized query.",
            "Re-run a Semgrep or SAST rule for SQL injection on the patched file.",
        ],
        "hint": "Security signal: attacker-controlled input reaches SQL query construction.",
    },
    "xss": {
        "owasp_top10_2021": "A03: Injection",
        "cwe_id": "CWE-79",
        "patch_id": "html_escape",
        "fix_family": "output_encoding",
        "patch_options": [
            {"id": "html_escape", "label": "Escape untrusted content before rendering"},
            {"id": "strip_script_substring", "label": "Remove only the literal script substring"},
            {"id": "disable_rendering", "label": "Disable the rendering path instead of fixing it"},
        ],
        "verification_plan": [
            "Render a script-tag payload and confirm it is treated as inert text.",
            "Verify legitimate user content still renders correctly after escaping.",
            "Re-run an XSS detection rule or regression test on the sink.",
        ],
        "hint": "Security signal: untrusted input reaches an HTML rendering sink without escaping.",
    },
    "broken_auth": {
        "owasp_top10_2021": "A07: Identification and Authentication Failures",
        "cwe_id": "CWE-287",
        "patch_id": "require_stronger_auth_check",
        "fix_family": "authentication_enforcement",
        "patch_options": [
            {"id": "require_stronger_auth_check", "label": "Enforce the missing authentication or session validation check"},
            {"id": "hide_entrypoint", "label": "Hide the route from the UI only"},
            {"id": "disable_feature", "label": "Disable the feature instead of fixing authentication"},
        ],
        "verification_plan": [
            "Attempt the protected flow with missing or invalid authentication and confirm it is rejected.",
            "Verify a legitimate authenticated user can still access the intended feature.",
            "Re-run authentication tests and any associated scanner rule.",
        ],
        "hint": "Security signal: the code allows a privileged path without sufficient authentication validation.",
    },
    "insecure_authentication": {
        "owasp_top10_2021": "A07: Identification and Authentication Failures",
        "cwe_id": "CWE-798",
        "patch_id": "remove_hardcoded_secret",
        "fix_family": "secret_externalization",
        "patch_options": [
            {"id": "remove_hardcoded_secret", "label": "Move the secret into managed configuration or a secret store"},
            {"id": "obfuscate_secret", "label": "Hide the secret with string splitting or encoding"},
            {"id": "disable_feature", "label": "Disable the feature instead of removing the secret"},
        ],
        "verification_plan": [
            "Confirm the secret is no longer present in the source tree or build artifact.",
            "Verify the application still authenticates correctly with the managed secret source.",
            "Re-run secret scanning and authentication smoke tests.",
        ],
        "hint": "Security signal: credentials or weak authentication material are embedded directly in code or config.",
    },
    "path_traversal": {
        "owasp_top10_2021": "A01: Broken Access Control",
        "cwe_id": "CWE-22",
        "patch_id": "validate_and_normalize_path",
        "fix_family": "path_canonicalization",
        "patch_options": [
            {"id": "validate_and_normalize_path", "label": "Normalize the path and enforce a bounded allowlisted root"},
            {"id": "strip_dotdot", "label": "Reject only dot-dot substrings without canonicalization"},
            {"id": "disable_file_access", "label": "Disable the file access feature entirely"},
        ],
        "verification_plan": [
            "Attempt ../ traversal payloads and confirm the resolved path cannot escape the safe root.",
            "Verify legitimate in-scope files are still accessible after the patch.",
            "Re-run traversal-focused tests or scanner rules.",
        ],
        "hint": "Security signal: user input controls filesystem path resolution without normalization and bounds checks.",
    },
    "file_inclusion": {
        "owasp_top10_2021": "A01: Broken Access Control",
        "cwe_id": "CWE-98",
        "patch_id": "allowlist_template_or_file",
        "fix_family": "file_allowlisting",
        "patch_options": [
            {"id": "allowlist_template_or_file", "label": "Resolve only files from a fixed allowlist or trusted directory map"},
            {"id": "strip_slashes_only", "label": "Strip slashes without validating the include target"},
            {"id": "disable_include_path", "label": "Disable inclusion rather than fixing the source selection"},
        ],
        "verification_plan": [
            "Attempt local or remote file inclusion payloads and confirm the include target is rejected.",
            "Verify legitimate templates or includes still load from the trusted set.",
            "Re-run file inclusion tests or scanner rules.",
        ],
        "hint": "Security signal: untrusted input controls which file or template is loaded.",
    },
    "command_injection": {
        "owasp_top10_2021": "A03: Injection",
        "cwe_id": "CWE-78",
        "patch_id": "remove_shell_true",
        "fix_family": "shell_elimination",
        "patch_options": [
            {"id": "remove_shell_true", "label": "Use a fixed executable and argument list without a shell"},
            {"id": "quote_input_only", "label": "Quote the user input but keep shell execution"},
            {"id": "disable_command", "label": "Disable the feature instead of removing shell injection"},
        ],
        "verification_plan": [
            "Replay shell metacharacter payloads and confirm no extra command executes.",
            "Verify the intended command still succeeds for safe input.",
            "Re-run command injection checks or regression tests.",
        ],
        "hint": "Security signal: user input reaches shell execution or command construction.",
    },
    "broken_access_control": {
        "owasp_top10_2021": "A01: Broken Access Control",
        "cwe_id": "CWE-862",
        "patch_id": "enforce_authorization_check",
        "fix_family": "server_side_authorization",
        "patch_options": [
            {"id": "enforce_authorization_check", "label": "Enforce a server-side authorization or ownership check"},
            {"id": "hide_identifier", "label": "Hide the object identifier from the client only"},
            {"id": "disable_route", "label": "Disable the route instead of fixing authorization"},
        ],
        "verification_plan": [
            "Attempt the request as an unauthorized user and confirm it is blocked.",
            "Verify the rightful owner or role still reaches the protected resource.",
            "Re-run authorization tests and SAST checks.",
        ],
        "hint": "Security signal: a privileged resource is accessible without a required authorization or ownership check.",
    },
    "insecure_deserialization": {
        "owasp_top10_2021": "A08: Software and Data Integrity Failures",
        "cwe_id": "CWE-502",
        "patch_id": "safe_serialization_format",
        "fix_family": "safe_serialization",
        "patch_options": [
            {"id": "safe_serialization_format", "label": "Replace unsafe object loading with a safe structured format and validation"},
            {"id": "wrap_serialized_blob", "label": "Encode the serialized blob but keep the same unsafe loader"},
            {"id": "disable_restore", "label": "Disable the feature instead of fixing deserialization"},
        ],
        "verification_plan": [
            "Replay a malicious serialized payload and confirm arbitrary object loading is blocked.",
            "Verify legitimate sessions or messages still deserialize with the safe format.",
            "Re-run deserialization-focused checks or tests.",
        ],
        "hint": "Security signal: untrusted serialized data is loaded into application objects.",
    },
    "csrf": {
        "owasp_top10_2021": "A01: Broken Access Control",
        "cwe_id": "CWE-352",
        "patch_id": "require_csrf_token",
        "fix_family": "anti_csrf_tokens",
        "patch_options": [
            {"id": "require_csrf_token", "label": "Require and validate an anti-CSRF token"},
            {"id": "check_referer_only", "label": "Validate only the Referer header"},
            {"id": "disable_form", "label": "Disable the form instead of fixing request forgery protection"},
        ],
        "verification_plan": [
            "Attempt a forged cross-site request and confirm the state change is rejected.",
            "Verify same-site legitimate requests with a valid token still succeed.",
            "Re-run request-forgery regression tests.",
        ],
        "hint": "Security signal: a state-changing route lacks an anti-CSRF control.",
    },
}

KEYWORD_PATTERNS = [
    ("sql_injection", ("sql injection", "select *", "where username", "database query", "query =")),
    ("xss", ("xss", "html output", "render_comment", "<script", "innerhtml", "render(")),
    ("broken_auth", ("login bypass", "auth bypass", "session check", "missing auth", "unauthenticated")),
    ("insecure_authentication", ("hardcoded credential", "hardcoded password", "api_key", "secret", "md5", "weak password")),
    ("path_traversal", ("path traversal", "../", "open(base", "filesystem path", "directory traversal")),
    ("file_inclusion", ("file inclusion", "include(", "template_name", "remote file", "local file inclusion")),
    ("command_injection", ("command injection", "os.system", "shell=true", "subprocess", "exec(")),
    ("broken_access_control", ("idor", "access control", "ownership check", "admin route", "authorize")),
    ("insecure_deserialization", ("pickle.loads", "deserialize", "yaml.load", "unsafe deserialization")),
    ("csrf", ("csrf", "forged request", "state-changing request", "missing csrf")),
]

CODE_FIELD_CANDIDATES = (
    "code_snippet",
    "code",
    "func",
    "function",
    "function_source",
    "source",
    "source_code",
    "processed_func",
    "before",
    "before_code",
    "vulnerable_code",
    "buggy",
    "snippet",
)
FIXED_CODE_FIELD_CANDIDATES = (
    "fixed_code",
    "patched_code",
    "after",
    "after_code",
    "fixed",
    "patch",
)
TEXT_FIELD_CANDIDATES = (
    "description",
    "summary",
    "message",
    "commit_message",
    "commit_msg",
    "title",
    "details",
    "scanner_hint",
    "cve_description",
    "path",
    "file_path",
    "filename",
)
BOOLEAN_FIELD_CANDIDATES = (
    "vulnerable",
    "is_vulnerable",
    "label",
    "target",
    "bug",
    "security",
    "is_bad",
)
DIRECTORY_CODE_SUFFIXES = {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".php", ".c", ".cpp", ".go", ".rb"}

SYNTHETIC_WEB_PATTERNS = [
    {
        "task_id": "synthetic_sqli_concat",
        "vulnerability_type": "sql_injection",
        "code_snippet": "def login(user, password):\n    sql = \"SELECT * FROM users WHERE user='\" + user + \"' AND password='\" + password + \"'\"\n    return db.execute(sql).fetchone()\n",
        "hint": "Scanner warning: dynamic SQL query built from user input.",
    },
    {
        "task_id": "synthetic_xss_renderer",
        "vulnerability_type": "xss",
        "code_snippet": "def show_comment(comment):\n    return f\"<div class='comment'>{comment}</div>\"\n",
        "hint": "Scanner warning: user-controlled HTML content is rendered without escaping.",
    },
    {
        "task_id": "synthetic_broken_auth_admin",
        "vulnerability_type": "broken_auth",
        "code_snippet": "def billing_admin(session):\n    if session.user:\n        return render_billing_admin()\n    abort(401)\n",
        "hint": "Scanner warning: any authenticated user can reach an admin-only route.",
    },
    {
        "task_id": "synthetic_path_traversal",
        "vulnerability_type": "path_traversal",
        "code_snippet": "def read_report(name):\n    return open('/srv/reports/' + name).read()\n",
        "hint": "Scanner warning: request data controls a filesystem path without validation.",
    },
    {
        "task_id": "synthetic_command_injection",
        "vulnerability_type": "command_injection",
        "code_snippet": "def archive(target):\n    os.system('tar -czf backup.tgz ' + target)\n",
        "hint": "Scanner warning: shell command concatenates user input.",
    },
    {
        "task_id": "synthetic_insecure_deser",
        "vulnerability_type": "insecure_deserialization",
        "code_snippet": "def restore(blob):\n    return pickle.loads(blob)\n",
        "hint": "Scanner warning: untrusted serialized bytes are loaded with pickle.",
    },
    {
        "task_id": "synthetic_csrf_profile",
        "vulnerability_type": "csrf",
        "code_snippet": "@app.post('/profile')\ndef update_profile(request):\n    current_user.update(request.form)\n",
        "hint": "Scanner warning: state-changing request lacks anti-CSRF protection.",
    },
    {
        "task_id": "synthetic_safe_query",
        "vulnerability_type": None,
        "code_snippet": "def get_user(conn, user_id):\n    return conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()\n",
        "hint": "Scanner note: query is parameterized and no obvious sink is exposed.",
    },
]


def normalize_vulnerability_label(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return None
    normalized = LABEL_ALIASES.get(normalized, normalized)
    return normalized if normalized in RELEVANT_VULNERABILITY_TYPES else None


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        return _read_directory_records(path)

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        records.append(payload)
        return records

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("records", "items", "data", "examples", "rows", "vulnerabilities"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [payload]
        return []

    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))

    raise ValueError(f"Unsupported external dataset format: {path}")


def _read_directory_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in DIRECTORY_CODE_SUFFIXES:
            continue
        code = _read_text_file(file_path)
        if not code.strip():
            continue
        relative_path = str(file_path.relative_to(root))
        filename_lower = file_path.name.lower()
        record: dict[str, Any] = {
            "path": relative_path,
            "filename": file_path.name,
            "code_snippet": code,
        }
        cwe_match = re.search(r"cwe[-_ ]?(\d+)", relative_path, re.IGNORECASE)
        if cwe_match:
            record["cwe_id"] = f"CWE-{cwe_match.group(1)}"
        if "good" in filename_lower and "bad" not in filename_lower:
            record["vulnerable"] = False
        elif "bad" in filename_lower:
            record["vulnerable"] = True
        records.append(record)
    return records


def _extract_string(record: dict[str, Any], fields: Iterable[str]) -> str:
    for field in fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_code(record: dict[str, Any]) -> str:
    code = _extract_string(record, CODE_FIELD_CANDIDATES)
    return code[:6000]


def _extract_fixed_code(record: dict[str, Any]) -> str:
    code = _extract_string(record, FIXED_CODE_FIELD_CANDIDATES)
    return code[:6000]


def _normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "bad", "vulnerable"}:
            return True
        if lowered in {"0", "false", "no", "n", "good", "safe", "benign"}:
            return False
    return None


def _infer_vulnerable(record: dict[str, Any]) -> bool | None:
    for field in BOOLEAN_FIELD_CANDIDATES:
        if field in record:
            value = _normalize_bool(record[field])
            if value is not None:
                return value
    return None


def _extract_cwe_tokens(record: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    for field in ("cwe", "cwe_id", "cwe_ids", "cwe_list", "weakness", "weakness_id"):
        value = record.get(field)
        if value is None:
            continue
        if isinstance(value, list):
            text = " ".join(str(item) for item in value)
        else:
            text = str(value)
        tokens.extend(re.findall(r"CWE[-_ ]?(\d+)", text, re.IGNORECASE))
    combined_text = " ".join(str(record.get(field, "")) for field in TEXT_FIELD_CANDIDATES)
    tokens.extend(re.findall(r"CWE[-_ ]?(\d+)", combined_text, re.IGNORECASE))
    return tokens


def _combined_record_text(record: dict[str, Any]) -> str:
    parts = [str(record.get(field, "")) for field in TEXT_FIELD_CANDIDATES]
    code = _extract_code(record)
    if code:
        parts.append(code[:1200])
    return " ".join(part for part in parts if part).lower()


def _infer_vulnerability_label(record: dict[str, Any]) -> str | None:
    for field in ("vulnerability_type", "vuln_type", "bug_type", "category"):
        label = normalize_vulnerability_label(record.get(field))
        if label:
            return label

    for cwe in _extract_cwe_tokens(record):
        mapped = CWE_TO_LABEL.get(cwe)
        if mapped:
            return mapped

    combined_text = _combined_record_text(record)
    for label, keywords in KEYWORD_PATTERNS:
        if any(keyword in combined_text for keyword in keywords):
            return label
    return None


def _looks_web_relevant(record: dict[str, Any], label: str | None) -> bool:
    if label:
        return True
    combined_text = _combined_record_text(record)
    return any(
        token in combined_text
        for token in (
            "sql",
            "html",
            "render",
            "login",
            "auth",
            "csrf",
            "path",
            "file",
            "command",
            "shell",
            "template",
            "secret",
            "cookie",
            "session",
        )
    )


def _metadata_for_label(label: str | None) -> dict[str, Any]:
    if label is None:
        return {}
    return VULNERABILITY_METADATA.get(label, {})


def _build_review(label: str | None, *, rationale: str, include_verification: bool = False) -> dict[str, Any]:
    metadata = _metadata_for_label(label)
    review: dict[str, Any] = {"rationale": rationale}
    for key in ("owasp_top10_2021", "cwe_id", "fix_family"):
        value = metadata.get(key)
        if value:
            review[key] = value
    if include_verification:
        verification_plan = metadata.get("verification_plan")
        if verification_plan:
            review["verification_plan"] = verification_plan
    return review


def _make_row(
    *,
    task_id: str,
    instruction: str,
    code_snippet: str,
    scanner_hint: str,
    source: str,
    example_family: str,
    output: dict[str, Any],
    patch_options: list[dict[str, str]] | None = None,
    evidence: list[str] | None = None,
) -> dict[str, Any]:
    input_block: dict[str, Any] = {
        "code_snippet": code_snippet,
        "scanner_hint": scanner_hint,
    }
    if patch_options:
        input_block["patch_options"] = patch_options
    if evidence:
        input_block["evidence"] = evidence

    return {
        "task_id": task_id,
        "instruction": instruction,
        "input": input_block,
        "output": output,
        "source": source,
        "example_family": example_family,
    }


def _build_examples_for_record(record: dict[str, Any], *, source_name: str, index: int) -> list[dict[str, Any]]:
    code_snippet = _extract_code(record)
    if not code_snippet:
        return []

    label = _infer_vulnerability_label(record)
    if not _looks_web_relevant(record, label):
        return []

    vulnerable = _infer_vulnerable(record)
    fixed_code = _extract_fixed_code(record)
    description = _extract_string(record, TEXT_FIELD_CANDIDATES)
    base_id = f"{source_name}_{index:05d}"

    if vulnerable is False:
        return [
            _make_row(
                task_id=f"{base_id}_inspect",
                instruction="Inspect the code and decide the next safe action.",
                code_snippet=code_snippet,
                scanner_hint=description or "This sample is treated as a safe or negative control case.",
                source=source_name,
                example_family="workflow_policy",
                output={
                    "action": {"action_type": "inspect"},
                    "review": {
                        "rationale": "The available evidence is either benign or insufficient to justify a stronger action.",
                    },
                },
            )
        ]

    if label is None:
        return []

    metadata = _metadata_for_label(label)
    scanner_hint = description or metadata.get("hint", "Security signal: suspicious code pattern detected.")
    evidence: list[str] = []
    if fixed_code:
        evidence.append("Reference fixed code excerpt:\n" + fixed_code[:1000])

    classify_output = {
        "action": {
            "action_type": "classify",
            "vulnerability_type": label,
        },
        "review": _build_review(
            label,
            rationale=description or metadata.get("hint", "The code matches a known vulnerable pattern."),
        ),
    }
    examples = [
        _make_row(
            task_id=f"{base_id}_classify",
            instruction="Inspect the code and classify the vulnerability type.",
            code_snippet=code_snippet,
            scanner_hint=scanner_hint,
            source=source_name,
            example_family="taxonomy_mapping",
            output=classify_output,
            evidence=evidence or None,
        )
    ]

    patch_id = metadata.get("patch_id")
    patch_options = metadata.get("patch_options")
    if patch_id and patch_options:
        examples.append(
            _make_row(
                task_id=f"{base_id}_patch",
                instruction="Choose the safest patch for the vulnerability.",
                code_snippet=code_snippet,
                scanner_hint=scanner_hint,
                source=source_name,
                example_family="patch_choice",
                patch_options=patch_options,
                evidence=evidence or None,
                output={
                    "action": {
                        "action_type": "apply_patch",
                        "vulnerability_type": label,
                        "patch_id": patch_id,
                    },
                    "review": _build_review(
                        label,
                        rationale=description or "Pick the safe fix that blocks the exploit without removing the feature.",
                    ),
                },
            )
        )
        examples.append(
            _make_row(
                task_id=f"{base_id}_verify",
                instruction=f"The patch '{patch_id}' has been selected. Plan the verification step.",
                code_snippet=code_snippet,
                scanner_hint=scanner_hint,
                source=source_name,
                example_family="verification_planning",
                patch_options=patch_options,
                evidence=evidence or None,
                output={
                    "action": {
                        "action_type": "verify",
                        "patch_id": patch_id,
                    },
                    "review": _build_review(
                        label,
                        rationale="Verification must prove the exploit is blocked and the intended behavior still works.",
                        include_verification=True,
                    ),
                },
            )
        )

    return examples


def build_rows_from_source(source_name: str, path: Path) -> list[dict[str, Any]]:
    records = _read_records(path)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        rows.extend(_build_examples_for_record(record, source_name=source_name, index=index))
    return rows


def build_synthetic_web_pattern_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in SYNTHETIC_WEB_PATTERNS:
        task_id = scenario["task_id"]
        label = scenario["vulnerability_type"]
        code_snippet = scenario["code_snippet"]
        scanner_hint = scenario["hint"]

        if label is None:
            rows.append(
                _make_row(
                    task_id=f"{task_id}_inspect",
                    instruction="Inspect the code and decide the next safe action.",
                    code_snippet=code_snippet,
                    scanner_hint=scanner_hint,
                    source="synthetic_web_patterns",
                    example_family="workflow_policy",
                    output={
                        "action": {"action_type": "inspect"},
                        "review": {
                            "rationale": "This example is intentionally safe, so a stronger action would be over-aggressive.",
                        },
                    },
                )
            )
            continue

        metadata = _metadata_for_label(label)
        rows.append(
            _make_row(
                task_id=f"{task_id}_classify",
                instruction="Inspect the code and classify the vulnerability type.",
                code_snippet=code_snippet,
                scanner_hint=scanner_hint,
                source="synthetic_web_patterns",
                example_family="taxonomy_mapping",
                output={
                    "action": {"action_type": "classify", "vulnerability_type": label},
                    "review": _build_review(label, rationale=metadata.get("hint", scanner_hint)),
                },
            )
        )
        rows.append(
            _make_row(
                task_id=f"{task_id}_patch",
                instruction="Choose the safest patch for the vulnerability.",
                code_snippet=code_snippet,
                scanner_hint=scanner_hint,
                source="synthetic_web_patterns",
                example_family="patch_choice",
                patch_options=metadata.get("patch_options"),
                output={
                    "action": {
                        "action_type": "apply_patch",
                        "vulnerability_type": label,
                        "patch_id": metadata["patch_id"],
                    },
                    "review": _build_review(
                        label,
                        rationale="Pick the defensive patch that fixes the bug while preserving the feature.",
                    ),
                },
            )
        )
        rows.append(
            _make_row(
                task_id=f"{task_id}_verify",
                instruction=f"The patch '{metadata['patch_id']}' has been selected. Plan the verification step.",
                code_snippet=code_snippet,
                scanner_hint=scanner_hint,
                source="synthetic_web_patterns",
                example_family="verification_planning",
                patch_options=metadata.get("patch_options"),
                output={
                    "action": {
                        "action_type": "verify",
                        "patch_id": metadata["patch_id"],
                    },
                    "review": _build_review(
                        label,
                        rationale="Verification should confirm exploit blocking and functionality preservation.",
                        include_verification=True,
                    ),
                },
            )
        )
    return rows


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, int] | int]:
    source_counts = Counter()
    vuln_counts = Counter()
    action_counts = Counter()
    for row in rows:
        source_counts[row.get("source", "unknown")] += 1
        output = row.get("output", {})
        action = output.get("action", output)
        action_type = str(action.get("action_type", "inspect"))
        action_counts[action_type] += 1
        vulnerability_type = normalize_vulnerability_label(action.get("vulnerability_type"))
        if vulnerability_type:
            vuln_counts[vulnerability_type] += 1
    return {
        "total_examples": len(rows),
        "source_breakdown": dict(source_counts),
        "vulnerability_type_counts": dict(vuln_counts),
        "action_type_counts": dict(action_counts),
    }


def _print_summary(summary: dict[str, dict[str, int] | int]) -> None:
    print(f"Built {summary['total_examples']} external WebSec examples")
    for label, counts in (
        ("Source breakdown", summary["source_breakdown"]),
        ("Action counts", summary["action_type_counts"]),
        ("Vulnerability counts", summary["vulnerability_type_counts"]),
    ):
        print(f"  {label}:")
        if not counts:
            print("    - none")
            continue
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"    - {key}: {value}")


def build_external_websec_corpus(
    *,
    output_path: Path | str,
    primevul_path: Path | str | None = None,
    devign_path: Path | str | None = None,
    juliet_path: Path | str | None = None,
    owasp_benchmark_path: Path | str | None = None,
    bigvul_path: Path | str | None = None,
    cvefixes_path: Path | str | None = None,
    github_security_fixes_path: Path | str | None = None,
    nvd_cve_path: Path | str | None = None,
    exploitdb_path: Path | str | None = None,
    include_synthetic_web_patterns: bool = True,
) -> list[dict[str, Any]]:
    source_paths = {
        "primevul": primevul_path,
        "devign": devign_path,
        "juliet": juliet_path,
        "owasp_benchmark": owasp_benchmark_path,
        "bigvul": bigvul_path,
        "cvefixes": cvefixes_path,
        "github_security_fix": github_security_fixes_path,
        "nvd_cve": nvd_cve_path,
        "exploitdb_selected": exploitdb_path,
    }

    rows: list[dict[str, Any]] = []
    for source_name, raw_path in source_paths.items():
        if raw_path is None:
            continue
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"{source_name} source not found: {path}")
        source_rows = build_rows_from_source(source_name, path)
        print(f"Added {len(source_rows)} rows from {source_name} ({path})")
        rows.extend(source_rows)

    if include_synthetic_web_patterns:
        synthetic_rows = build_synthetic_web_pattern_rows()
        print(f"Added {len(synthetic_rows)} synthetic web-pattern rows")
        rows.extend(synthetic_rows)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    _print_summary(_summarize_rows(rows))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a normalized external WebSec JSONL corpus")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--primevul", help="PrimeVul export path (.jsonl/.json/.csv)")
    parser.add_argument("--devign", help="Devign export path (.jsonl/.json/.csv)")
    parser.add_argument("--juliet", help="Juliet/SARD export path or source directory")
    parser.add_argument("--owasp-benchmark", dest="owasp_benchmark", help="OWASP Benchmark export path or source directory")
    parser.add_argument("--bigvul", help="Big-Vul export path (.csv/.json/.jsonl)")
    parser.add_argument("--cvefixes", help="CVEfixes export path (.csv/.json/.jsonl)")
    parser.add_argument("--github-security-fixes", dest="github_security_fixes", help="Curated GitHub security fix pairs path")
    parser.add_argument("--nvd-cve", dest="nvd_cve", help="Curated NVD/CVE snippet export path")
    parser.add_argument("--exploitdb-selected", dest="exploitdb_selected", help="Curated defensive ExploitDB sample path")
    parser.add_argument(
        "--include-synthetic-web-patterns",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include built-in synthetic web vulnerability patterns (default: True)",
    )

    args = parser.parse_args()
    build_external_websec_corpus(
        output_path=args.output,
        primevul_path=args.primevul,
        devign_path=args.devign,
        juliet_path=args.juliet,
        owasp_benchmark_path=args.owasp_benchmark,
        bigvul_path=args.bigvul,
        cvefixes_path=args.cvefixes,
        github_security_fixes_path=args.github_security_fixes,
        nvd_cve_path=args.nvd_cve,
        exploitdb_path=args.exploitdb_selected,
        include_synthetic_web_patterns=args.include_synthetic_web_patterns,
    )


if __name__ == "__main__":
    main()
