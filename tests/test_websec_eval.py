"""Tests for JSONL ingestion and WebSec evaluation metrics."""

from __future__ import annotations

import json

from websec_repair_env.eval.websec_eval import compute_metrics, load_eval_cases
from websec_repair_env.training.data_loader import load_jsonl_dataset, summarize_dataset


def test_load_jsonl_dataset_supports_nested_and_legacy_outputs(tmp_path) -> None:
    dataset_path = tmp_path / "train.jsonl"
    rows = [
        {
            "task_id": "xss_case",
            "instruction": "Inspect the code and classify the vulnerability type.",
            "input": {
                "code_snippet": "return f'<div>{comment}</div>'",
                "scanner_hint": "HTML rendering sink without escaping.",
            },
            "output": {
                "action": {
                    "action_type": "classify",
                    "vulnerability_type": "xss",
                },
                "review": {
                    "owasp_top10_2021": "A03:2021-Injection",
                    "cwe_id": "CWE-79",
                    "rationale": "Untrusted HTML output.",
                },
            },
        },
        {
            "task_id": "sqli_patch_case",
            "instruction": "Choose the safest patch for the vulnerability.",
            "input": {
                "code_snippet": "query = 'SELECT * FROM users WHERE id=' + user_id",
                "scanner_hint": "Patch options: parameterized_query | strip_quotes",
            },
            "output": {
                "action_type": "apply_patch",
                "patch_id": "parameterized_query",
            },
        },
    ]
    dataset_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    dataset = load_jsonl_dataset(dataset_path)

    assert len(dataset) == 2
    assert dataset["action_type"] == ["classify", "apply_patch"]
    for response in dataset["response"]:
        payload = json.loads(response)
        assert "action" in payload
        assert payload["action"]["action_type"] in {"classify", "apply_patch"}


def test_summarize_dataset_reports_counts(tmp_path) -> None:
    dataset_path = tmp_path / "train.jsonl"
    rows = [
        {
            "task_id": "case1",
            "instruction": "Inspect the code and classify the vulnerability type.",
            "input": {"code_snippet": "bad sql", "scanner_hint": "SQL injection"},
            "output": {"action": {"action_type": "classify", "vulnerability_type": "sql_injection"}},
        },
        {
            "task_id": "case2",
            "instruction": "Inspect the code and classify the vulnerability type.",
            "input": {"code_snippet": "bad html", "scanner_hint": "XSS"},
            "output": {"action": {"action_type": "classify", "vulnerability_type": "xss"}},
        },
        {
            "task_id": "case3",
            "instruction": "Choose the safest patch for the vulnerability.",
            "input": {"code_snippet": "bad auth", "scanner_hint": "Patch options: require_admin_role"},
            "output": {"action": {"action_type": "apply_patch", "patch_id": "require_admin_role"}},
        },
    ]
    dataset_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    dataset = load_jsonl_dataset(dataset_path)
    summary = summarize_dataset(dataset)

    assert summary["total_examples"] == 3
    assert summary["source_breakdown"]["websec_jsonl"] == 3
    assert summary["action_type_counts"]["classify"] == 2
    assert summary["action_type_counts"]["apply_patch"] == 1
    assert summary["vulnerability_type_counts"]["sql_injection"] == 1
    assert summary["vulnerability_type_counts"]["xss"] == 1


def test_load_eval_cases_reads_expected_schema(tmp_path) -> None:
    cases_path = tmp_path / "eval.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "task_id": "safe_case",
                "category": "negative",
                "instruction": "Inspect the code and choose the next action.",
                "input": {
                    "code_snippet": "return conn.execute('SELECT 1')",
                    "scanner_hint": "No obvious issue.",
                },
                "expected": {"action_type": "inspect"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_eval_cases(cases_path)

    assert len(cases) == 1
    assert cases[0]["expected"]["action_type"] == "inspect"


def test_compute_metrics_tracks_dual_json_and_env_action_rates() -> None:
    metrics = compute_metrics(
        [
            {
                "expected": {"action_type": "classify", "vulnerability_type": "xss"},
                "predicted_action": {"action_type": "classify", "vulnerability_type": "xss"},
                "valid_dual_json": True,
                "valid_env_action": True,
            },
            {
                "expected": {"action_type": "apply_patch", "patch_id": "parameterized_query"},
                "predicted_action": {"action_type": "apply_patch", "patch_id": "strip_quotes"},
                "valid_dual_json": False,
                "valid_env_action": True,
            },
        ]
    )

    assert metrics["total_cases"] == 2
    assert metrics["action_type_accuracy"] == 1.0
    assert metrics["vulnerability_type_accuracy"] == 1.0
    assert metrics["patch_id_accuracy"] == 0.0
    assert metrics["valid_dual_json_rate"] == 0.5
    assert metrics["valid_env_action_extraction_rate"] == 1.0
