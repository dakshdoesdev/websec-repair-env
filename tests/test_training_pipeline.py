"""Tests for the WebSec training data and inference pipeline."""

from __future__ import annotations

import json
from collections import Counter

from datasets import Dataset

from websec_repair_env.training import data_loader
from websec_repair_env.training.inference import extract_env_action, parse_model_output


def _make_source_dataset(source: str, count: int, action_type: str = "classify") -> Dataset:
    return Dataset.from_dict(
        {
            "instruction": [f"{source} instruction {index}" for index in range(count)],
            "response": [
                json.dumps({"action": {"action_type": action_type}})
                for _ in range(count)
            ],
            "task_id": [f"{source}_{index}" for index in range(count)],
            "action_type": [action_type] * count,
            "source": [source] * count,
            "example_family": ["workflow_policy"] * count,
        }
    )


def test_report_converter_generates_all_example_families(tmp_path) -> None:
    research_doc = tmp_path / "research.md"
    research_doc.write_text(
        "\n".join(
            [
                "OWASP Top 10",
                "MITRE CWE and CVSS v3.1 are required.",
                "Use Semgrep, Trivy, and OWASP ZAP.",
                "Follow inspect -> classify -> apply_patch -> verify -> submit.",
                "Cover SQL injection, XSS, broken auth, SSRF, IDOR, path traversal.",
            ]
        ),
        encoding="utf-8",
    )

    dataset = data_loader.create_report_training_data(research_doc)

    assert 60 <= len(dataset) <= 120
    assert set(dataset["example_family"]) == {
        "taxonomy_mapping",
        "patch_choice",
        "verification_planning",
        "workflow_policy",
    }


def test_report_converter_emits_valid_dual_json(tmp_path) -> None:
    research_doc = tmp_path / "research.md"
    research_doc.write_text(
        "OWASP CWE CVSS Semgrep Trivy ZAP inspect classify patch verify submit",
        encoding="utf-8",
    )

    dataset = data_loader.create_report_training_data(research_doc)

    for response in dataset["response"]:
        payload = json.loads(response)
        assert "action" in payload
        assert payload["action"]["action_type"] in data_loader.ACTION_TYPES
        if "review" in payload:
            review = payload["review"]
            if "cvss_v3_1" in review:
                assert isinstance(review["cvss_v3_1"]["base_score"], float)
                assert isinstance(review["cvss_v3_1"]["vector"], str)
            if "verification_plan" in review:
                assert isinstance(review["verification_plan"], list)


def test_create_combined_dataset_balances_report_and_swe(monkeypatch) -> None:
    monkeypatch.setattr(
        data_loader,
        "load_trajectory_logs",
        lambda path: _make_source_dataset("trajectory", 250),
    )
    monkeypatch.setattr(
        data_loader,
        "create_synthetic_training_data",
        lambda: _make_source_dataset("synthetic", 6),
    )
    monkeypatch.setattr(
        data_loader,
        "create_report_training_data",
        lambda path: _make_source_dataset("research_doc", 60),
    )
    monkeypatch.setattr(
        data_loader,
        "load_swe_bench_lite",
        lambda: _make_source_dataset("swe_bench_lite", 534, action_type="apply_patch"),
    )

    dataset = data_loader.create_combined_dataset(
        trajectory_logs_path="trajectory_logs.json",
        research_doc_path="research.md",
        include_swe_bench=True,
        include_synthetic=True,
    )

    counts = Counter(dataset["source"])
    total = len(dataset)

    assert counts["research_doc"] / total >= 0.30
    assert counts["swe_bench_lite"] / total <= 0.25


def test_parse_model_output_prefers_dual_json() -> None:
    payload = parse_model_output(
        'some prompt text {"action": {"action_type": "classify", "vulnerability_type": "xss"}, '
        '"review": {"owasp_top10_2021": "A03: Injection", "cwe_id": "CWE-79", '
        '"rationale": "User input reaches HTML output without escaping."}}'
    )

    assert payload["action"]["action_type"] == "classify"
    assert payload["action"]["vulnerability_type"] == "xss"
    assert payload["review"]["cwe_id"] == "CWE-79"
    assert payload["action_type"] == "classify"
    assert payload["vulnerability_type"] == "xss"


def test_parse_model_output_supports_legacy_flat_json() -> None:
    payload = parse_model_output(
        'legacy response {"action_type": "apply_patch", "patch_id": "parameterized_query"}'
    )

    assert payload["action"]["action_type"] == "apply_patch"
    assert payload["action"]["patch_id"] == "parameterized_query"
    assert payload["patch_id"] == "parameterized_query"


def test_extract_env_action_keeps_current_env_meaning() -> None:
    dual_payload = {
        "action": {
            "action_type": "apply_patch",
            "vulnerability_type": "sql_injection",
            "patch_id": "parameterized_query",
        },
        "review": {
            "owasp_top10_2021": "A03: Injection",
            "cwe_id": "CWE-89",
            "rationale": "Use bound parameters.",
        },
    }

    env_action = extract_env_action(dual_payload)

    assert env_action == {
        "action_type": "apply_patch",
        "vulnerability_type": "sql_injection",
        "patch_id": "parameterized_query",
    }
