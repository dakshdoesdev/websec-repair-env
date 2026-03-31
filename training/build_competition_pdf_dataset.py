#!/usr/bin/env python3
"""Build WebSec training JSONL from the competition PDF dataset.

The PDF in ``COMPETEION INSTRCUTONS + MICELL/DATA.pdf`` contains repeated
JSON-like records for web security vulnerabilities. This script extracts those
records, normalizes the wrapped PDF text, and emits the repo's existing JSONL
training schema so the output can be passed directly to ``training/qlora_train.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


STRING_PATTERN = r'"((?:\\.|[^"\\])*)"'
RECORD_PATTERN = re.compile(
    rf'"category":\s*{STRING_PATTERN},\s*'
    rf'"vulnerability":\s*{STRING_PATTERN},\s*'
    rf'"cwe_id":\s*{STRING_PATTERN},\s*'
    rf'"severity":\s*{STRING_PATTERN},\s*'
    rf'"description":\s*{STRING_PATTERN},\s*'
    rf'"insecure_code":\s*{STRING_PATTERN},\s*'
    rf'"secure_code":\s*{STRING_PATTERN},\s*'
    rf'"fix":\s*\[(.*?)\]\s*\}}',
    re.S,
)
FIX_ITEM_PATTERN = re.compile(STRING_PATTERN, re.S)


@dataclass(frozen=True)
class VulnerabilityConfig:
    vulnerability_type: str
    owasp_top10_2021: str
    fix_family: str
    patch_id: str
    patch_label: str
    verification_plan: tuple[str, ...]


VULNERABILITY_CONFIG: dict[str, VulnerabilityConfig] = {
    "SQL Injection": VulnerabilityConfig(
        vulnerability_type="sql_injection",
        owasp_top10_2021="A03:2021-Injection",
        fix_family="parameterized_queries",
        patch_id="use_parameterized_queries",
        patch_label="Replace string-built SQL with parameterized queries",
        verification_plan=(
            "Replay quote-based and boolean-based SQL injection payloads and confirm the query is not altered.",
            "Run the normal success path to confirm valid credentials or search terms still work.",
            "Re-run the SQL injection scanner or regression tests on the patched handler.",
        ),
    ),
    "Command Injection": VulnerabilityConfig(
        vulnerability_type="command_injection",
        owasp_top10_2021="A03:2021-Injection",
        fix_family="safe_subprocess_invocation",
        patch_id="remove_shell_execution",
        patch_label="Replace shell execution with a safe argument list and input allowlist",
        verification_plan=(
            "Try shell metacharacter payloads and confirm they are rejected or treated as plain data.",
            "Run the intended command with a normal input and confirm the expected result still succeeds.",
            "Re-run command-injection checks to confirm the shell invocation path is removed.",
        ),
    ),
    "Path Traversal": VulnerabilityConfig(
        vulnerability_type="path_traversal",
        owasp_top10_2021="A01:2021-Broken Access Control",
        fix_family="path_canonicalization",
        patch_id="validate_and_normalize_path",
        patch_label="Normalize the path and enforce a base-directory allowlist",
        verification_plan=(
            "Try '../' traversal payloads and confirm access outside the allowed directory is blocked.",
            "Fetch a legitimate in-scope file and confirm the normal download or read flow still works.",
            "Verify the resolved canonical path always stays within the intended base directory.",
        ),
    ),
    "Server-Side Request Forgery": VulnerabilityConfig(
        vulnerability_type="ssrf",
        owasp_top10_2021="A10:2021-Server-Side Request Forgery (SSRF)",
        fix_family="outbound_request_allowlisting",
        patch_id="restrict_outbound_requests",
        patch_label="Restrict outbound requests with URL validation and network allowlisting",
        verification_plan=(
            "Attempt requests to localhost, metadata, and RFC1918 targets and confirm they are blocked.",
            "Verify an approved public destination still works through the intended workflow.",
            "Confirm the URL parser, DNS resolution, and egress policy all enforce the same allowlist.",
        ),
    ),
    "Cross-Site Request Forgery": VulnerabilityConfig(
        vulnerability_type="csrf",
        owasp_top10_2021="A01:2021-Broken Access Control",
        fix_family="csrf_token_validation",
        patch_id="require_csrf_token",
        patch_label="Require a CSRF token and reject state-changing requests without it",
        verification_plan=(
            "Submit the action without a CSRF token and confirm the request is rejected.",
            "Submit the same action with a valid session and CSRF token and confirm it succeeds.",
            "Confirm cookies use the expected SameSite and secure settings for the endpoint.",
        ),
    ),
    "Hardcoded Secrets": VulnerabilityConfig(
        vulnerability_type="insecure_authentication",
        owasp_top10_2021="A07:2021-Identification and Authentication Failures",
        fix_family="secret_externalization",
        patch_id="remove_hardcoded_secret",
        patch_label="Move secrets out of source code into managed configuration or secret storage",
        verification_plan=(
            "Confirm the application refuses to start when required secrets are missing from the environment or secret manager.",
            "Verify the secret values no longer appear in the source tree, images, or logs.",
            "Rotate the exposed credentials and confirm the application works with the new secret source.",
        ),
    ),
    "Weak Password Storage": VulnerabilityConfig(
        vulnerability_type="insecure_authentication",
        owasp_top10_2021="A07:2021-Identification and Authentication Failures",
        fix_family="modern_password_hashing",
        patch_id="use_modern_password_hash",
        patch_label="Replace weak password storage with bcrypt, scrypt, or argon2",
        verification_plan=(
            "Create a new credential and confirm the stored value is a salted password hash, not a plaintext or fast hash.",
            "Verify a correct password still authenticates and an incorrect one is rejected.",
            "Confirm any migration path handles legacy passwords safely without breaking login.",
        ),
    ),
    "Broken Access Control / IDOR": VulnerabilityConfig(
        vulnerability_type="broken_access_control",
        owasp_top10_2021="A01:2021-Broken Access Control",
        fix_family="object_level_authorization",
        patch_id="enforce_object_ownership",
        patch_label="Enforce per-object authorization before returning or updating data",
        verification_plan=(
            "Try another user's object identifier and confirm access is denied or returns 404.",
            "Verify the legitimate owner can still read or modify the resource successfully.",
            "Re-run authorization tests to confirm object ownership is enforced at the query or service layer.",
        ),
    ),
}


def _extract_pdf_text(pdf_path: Path) -> str:
    if shutil.which("pdftotext"):
        return subprocess.check_output(["pdftotext", str(pdf_path), "-"], text=True)

    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on runtime tooling
        raise RuntimeError(
            "Could not extract PDF text. Install poppler-utils for pdftotext or `pip install pypdf`."
        ) from exc

    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _decode_wrapped_json_string(raw_value: str) -> str:
    normalized = raw_value.replace("\f", "\n").replace("\r", "\n")
    normalized = normalized.replace("\n", " ")
    normalized = re.sub(r"[ \t]+", " ", normalized).strip()
    # Some PDF wraps split escapes like "\n" into "\ n"; stitch them back together.
    normalized = re.sub(r'\\\s+([nrt"\'\\/])', r"\\\1", normalized)
    return json.loads(f'"{normalized}"')


def _clean_code_block(raw_value: str) -> str:
    decoded = _decode_wrapped_json_string(raw_value)
    cleaned_lines: list[str] = []
    for line in decoded.splitlines():
        if not line.strip():
            cleaned_lines.append("")
            continue
        leading = re.match(r"^\s*", line).group(0)
        content = re.sub(r"\s+", " ", line.strip())
        cleaned_lines.append(f"{leading}{content}".rstrip())
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _clean_text_block(raw_value: str) -> str:
    text = _decode_wrapped_json_string(raw_value)
    return re.sub(r"\s+", " ", text).strip()


def _redact_secrets(code: str) -> str:
    """Replace API keys and secrets with placeholder values to avoid GitHub secret scanning."""
    # Stripe API keys
    code = re.sub(r'sk_live_[a-zA-Z0-9]{24}', 'sk_test_PLACEHOLDER_STRIPE_API_KEY', code)
    code = re.sub(r'sk_test_[a-zA-Z0-9]{24}', 'sk_test_PLACEHOLDER_STRIPE_API_KEY', code)
    
    # Webhook signing secrets
    code = re.sub(r'wh_sk_live_[a-zA-Z0-9]+', 'wh_sk_test_PLACEHOLDER_WEBHOOK_SECRET', code)
    
    # Twilio Account SIDs
    code = re.sub(r'AC[a-f0-9]{32}', 'AC_PLACEHOLDER_TWILIO_ACCOUNT_SID', code)
    
    # Generic API keys (long alphanumeric strings that look like keys)
    code = re.sub(r'\b[a-f0-9]{32}\b', 'PLACEHOLDER_API_KEY_32_CHARS', code)
    code = re.sub(r'\b[a-f0-9]{40}\b', 'PLACEHOLDER_API_KEY_40_CHARS', code)
    
    # Password patterns
    code = re.sub(r'Sup3rS3cr3tPa\$\$word!', '[REDACTED_DB_PASSWORD]', code)
    
    return code


def _slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _truncate_sentence(value: str, limit: int = 220) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _parse_records(text: str) -> list[dict[str, object]]:
    text = text.replace("\f", "\n")
    records: list[dict[str, object]] = []

    for match in RECORD_PATTERN.finditer(text):
        vulnerability = _clean_text_block(match.group(2))
        if vulnerability not in VULNERABILITY_CONFIG:
            raise ValueError(f"Unsupported vulnerability label in PDF: {vulnerability}")

        fixes = [_clean_text_block(item) for item in FIX_ITEM_PATTERN.findall(match.group(8))]
        records.append(
            {
                "category": _clean_text_block(match.group(1)),
                "vulnerability": vulnerability,
                "cwe_id": _clean_text_block(match.group(3)),
                "severity": _clean_text_block(match.group(4)),
                "description": _clean_text_block(match.group(5)),
                "insecure_code": _clean_code_block(match.group(6)),
                "secure_code": _clean_code_block(match.group(7)),
                "fix": fixes,
            }
        )

    return records


def _build_review_payload(
    *,
    config: VulnerabilityConfig,
    cwe_id: str,
    rationale: str,
    verification_plan: list[str] | None = None,
) -> dict[str, object]:
    review: dict[str, object] = {
        "owasp_top10_2021": config.owasp_top10_2021,
        "cwe_id": cwe_id,
        "fix_family": config.fix_family,
        "rationale": rationale,
    }
    if verification_plan:
        review["verification_plan"] = verification_plan
    return review


def _build_training_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for index, record in enumerate(records, start=1):
        vulnerability = str(record["vulnerability"])
        config = VULNERABILITY_CONFIG[vulnerability]
        vuln_slug = _slugify(config.vulnerability_type)
        cwe_id = str(record["cwe_id"])
        severity = str(record["severity"])
        description = str(record["description"])
        insecure_code = _redact_secrets(str(record["insecure_code"]))
        secure_code = str(record["secure_code"])
        fixes = list(record["fix"])
        base_task_id = f"competition_{vuln_slug}_{index:03d}"
        scanner_hint = (
            f"Competition PDF case: {vulnerability} ({cwe_id}, {severity}). "
            f"{_truncate_sentence(description)}"
        )
        common_evidence = [
            f"Severity: {severity}",
            f"Primary fixes: {'; '.join(str(item) for item in fixes[:3])}",
        ]

        rows.append(
            {
                "task_id": f"{base_task_id}_classify",
                "instruction": "Inspect the code and classify the vulnerability type.",
                "input": {
                    "code_snippet": insecure_code,
                    "scanner_hint": scanner_hint,
                    "evidence": common_evidence,
                },
                "output": {
                    "action": {
                        "action_type": "classify",
                        "vulnerability_type": config.vulnerability_type,
                    },
                    "review": _build_review_payload(
                        config=config,
                        cwe_id=cwe_id,
                        rationale=description,
                    ),
                },
                "source": "competition_pdf",
                "example_family": "taxonomy_mapping",
            }
        )

        rows.append(
            {
                "task_id": f"{base_task_id}_patch",
                "instruction": (
                    "Choose the safest remediation pattern and apply the patch family that fixes the root cause."
                ),
                "input": {
                    "code_snippet": insecure_code,
                    "scanner_hint": scanner_hint,
                    "patch_options": [
                        {"id": config.patch_id, "label": config.patch_label},
                        {"id": "partial_input_filter", "label": "Add superficial input filtering only"},
                        {"id": "log_and_ignore", "label": "Log the issue but keep the vulnerable behavior"},
                    ],
                    "episode_state": (
                        f"Workflow stage: classify complete\n"
                        f"Known vulnerability: {config.vulnerability_type}\n"
                        "Secure reference implementation:\n"
                        f"```\n{secure_code}\n```"
                    ),
                    "evidence": common_evidence,
                },
                "output": {
                    "action": {
                        "action_type": "apply_patch",
                        "vulnerability_type": config.vulnerability_type,
                        "patch_id": config.patch_id,
                    },
                    "review": _build_review_payload(
                        config=config,
                        cwe_id=cwe_id,
                        rationale=(
                            f"The secure reference implementation removes the root cause of {vulnerability.lower()} "
                            "instead of adding a superficial check."
                        ),
                    ),
                },
                "source": "competition_pdf",
                "example_family": "patch_choice",
            }
        )

        rows.append(
            {
                "task_id": f"{base_task_id}_verify",
                "instruction": "Propose the next verification step after the patch is applied.",
                "input": {
                    "code_snippet": secure_code,
                    "scanner_hint": f"Validate the remediation for {vulnerability}.",
                    "episode_state": (
                        f"Workflow stage: patch applied\n"
                        f"Patched vulnerability: {config.vulnerability_type}\n"
                        f"Selected patch family: {config.fix_family}"
                    ),
                    "evidence": common_evidence,
                },
                "output": {
                    "action": {
                        "action_type": "verify",
                        "vulnerability_type": config.vulnerability_type,
                    },
                    "review": _build_review_payload(
                        config=config,
                        cwe_id=cwe_id,
                        rationale="Verification must prove the exploit path is blocked without breaking the intended behavior.",
                        verification_plan=list(config.verification_plan),
                    ),
                },
                "source": "competition_pdf",
                "example_family": "verification_planning",
            }
        )

    return rows


def build_dataset(pdf_path: Path, output_path: Path) -> int:
    records = _parse_records(_extract_pdf_text(pdf_path))
    rows = _build_training_rows(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")

    print(f"Parsed {len(records)} competition PDF records")
    print(f"Wrote {len(rows)} training rows to {output_path}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, help="Path to DATA.pdf")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    args = parser.parse_args()

    build_dataset(Path(args.pdf), Path(args.output))


if __name__ == "__main__":
    main()
