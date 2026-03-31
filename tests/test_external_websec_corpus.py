"""Tests for building and balancing the external WebSec corpus."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter

from datasets import Dataset

from websec_repair_env.training import data_loader
from websec_repair_env.training.build_external_websec_corpus import (
    build_external_websec_corpus,
)


def _make_source_dataset(source: str, count: int) -> Dataset:
    return Dataset.from_dict(
        {
            "instruction": [f"{source} instruction {index}" for index in range(count)],
            "response": [
                json.dumps({"action": {"action_type": "classify", "vulnerability_type": "xss"}})
                for _ in range(count)
            ],
            "task_id": [f"{source}_{index}" for index in range(count)],
            "action_type": ["classify"] * count,
            "source": [source] * count,
            "example_family": ["taxonomy_mapping"] * count,
        }
    )


def test_build_external_websec_corpus_normalizes_mixed_sources(tmp_path) -> None:
    primevul_path = tmp_path / "primevul.jsonl"
    primevul_path.write_text(
        json.dumps(
            {
                "code": "query = \"SELECT * FROM users WHERE name='\" + username + \"'\"",
                "cwe_id": "CWE-89",
                "description": "PrimeVul example: dynamic SQL from user input",
                "vulnerable": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    devign_path = tmp_path / "devign.json"
    devign_path.write_text(
        json.dumps(
            [
                {
                    "function": "def render(comment):\n    return f\"<div>{comment}</div>\"\n",
                    "cwe": "CWE-79",
                    "target": 1,
                    "message": "Devign example: unsafe HTML rendering sink",
                }
            ]
        ),
        encoding="utf-8",
    )

    juliet_dir = tmp_path / "juliet"
    juliet_dir.mkdir()
    (juliet_dir / "CWE22_Path_Traversal__bad.py").write_text(
        "def read(name):\n    return open('/srv/data/' + name).read()\n",
        encoding="utf-8",
    )

    owasp_path = tmp_path / "owasp_benchmark.jsonl"
    owasp_path.write_text(
        json.dumps(
            {
                "code_snippet": "return conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()",
                "cwe_id": "CWE-89",
                "description": "OWASP Benchmark safe control case with parameterized query",
                "vulnerable": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    bigvul_path = tmp_path / "bigvul.csv"
    with bigvul_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["before", "after", "cwe_id", "description", "vulnerable"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "before": "subprocess.run(f'tar -czf backup.tgz {target}', shell=True)",
                "after": "subprocess.run(['tar', '-czf', 'backup.tgz', target], shell=False)",
                "cwe_id": "CWE-78",
                "description": "Big-Vul patch pair for shell command injection",
                "vulnerable": "1",
            }
        )

    github_fixes_path = tmp_path / "github_security_fixes.jsonl"
    github_fixes_path.write_text(
        json.dumps(
            {
                "before_code": "API_KEY = 'prod-live-secret'\nclient = PaymentsClient(api_key=API_KEY)\n",
                "fixed_code": "API_KEY = os.environ['PAYMENTS_API_KEY']\nclient = PaymentsClient(api_key=API_KEY)\n",
                "cwe_id": "CWE-798",
                "description": "GitHub security fix removing a hardcoded production secret",
                "vulnerable": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "external_websec.jsonl"
    build_external_websec_corpus(
        output_path=output_path,
        primevul_path=primevul_path,
        devign_path=devign_path,
        juliet_path=juliet_dir,
        owasp_benchmark_path=owasp_path,
        bigvul_path=bigvul_path,
        github_security_fixes_path=github_fixes_path,
        include_synthetic_web_patterns=False,
    )

    dataset = data_loader.load_jsonl_dataset(output_path, preserve_row_source=True)
    summary = data_loader.summarize_dataset(dataset)

    assert summary["source_breakdown"]["primevul"] >= 1
    assert summary["source_breakdown"]["devign"] >= 1
    assert summary["source_breakdown"]["juliet"] >= 1
    assert summary["source_breakdown"]["owasp_benchmark"] >= 1
    assert summary["source_breakdown"]["bigvul"] >= 1
    assert summary["source_breakdown"]["github_security_fix"] >= 1
    assert summary["vulnerability_type_counts"]["sql_injection"] >= 1
    assert summary["vulnerability_type_counts"]["xss"] >= 1
    assert summary["vulnerability_type_counts"]["path_traversal"] >= 1
    assert summary["vulnerability_type_counts"]["command_injection"] >= 1
    assert summary["vulnerability_type_counts"]["insecure_authentication"] >= 1
    assert summary["action_type_counts"]["inspect"] >= 1
    assert summary["action_type_counts"]["apply_patch"] >= 1
    assert summary["action_type_counts"]["verify"] >= 1


def test_build_external_websec_corpus_adds_synthetic_patterns(tmp_path) -> None:
    output_path = tmp_path / "synthetic_websec.jsonl"
    rows = build_external_websec_corpus(
        output_path=output_path,
        include_synthetic_web_patterns=True,
    )

    dataset = data_loader.load_jsonl_dataset(output_path, preserve_row_source=True)
    summary = data_loader.summarize_dataset(dataset)

    assert rows
    assert summary["source_breakdown"]["synthetic_web_patterns"] >= 1
    assert summary["vulnerability_type_counts"]["sql_injection"] >= 1
    assert summary["vulnerability_type_counts"]["csrf"] >= 1
    assert summary["vulnerability_type_counts"]["insecure_deserialization"] >= 1
    assert summary["action_type_counts"]["inspect"] >= 1


def test_rebalance_external_dataset_caps_groups_against_custom_data() -> None:
    external = data_loader._concat_datasets(
        [
            _make_source_dataset("primevul", 200),
            _make_source_dataset("devign", 200),
            _make_source_dataset("juliet", 200),
            _make_source_dataset("owasp_benchmark", 200),
            _make_source_dataset("bigvul", 200),
            _make_source_dataset("github_security_fix", 200),
        ]
    )

    rebalanced = data_loader._rebalance_external_dataset(external, custom_dataset_size=350)
    counts = Counter(rebalanced["source"])
    desired_total = max(
        350,
        math.ceil(350 / data_loader.SOURCE_GROUP_TARGET_SHARES["custom"]),
    )
    expected_primevul_devign = math.ceil(
        desired_total * data_loader.SOURCE_GROUP_TARGET_SHARES["primevul_devign"]
    )
    expected_juliet_owasp = math.ceil(
        desired_total * data_loader.SOURCE_GROUP_TARGET_SHARES["juliet_owasp"]
    )
    expected_patch_pairs = math.ceil(
        desired_total * data_loader.SOURCE_GROUP_TARGET_SHARES["patch_pairs"]
    )

    assert counts["primevul"] + counts["devign"] == expected_primevul_devign
    assert counts["juliet"] + counts["owasp_benchmark"] == expected_juliet_owasp
    assert counts["bigvul"] + counts["github_security_fix"] == expected_patch_pairs
    assert counts["primevul"] == 200
    assert counts["devign"] == expected_primevul_devign - 200


def test_rebalance_external_dataset_keeps_custom_like_external_rows() -> None:
    external = data_loader._concat_datasets(
        [
            _make_source_dataset("synthetic_web_patterns", 22),
            _make_source_dataset("primevul", 20),
        ]
    )

    rebalanced = data_loader._rebalance_external_dataset(external, custom_dataset_size=200)
    counts = Counter(rebalanced["source"])

    assert counts["synthetic_web_patterns"] == 22
    assert counts["primevul"] >= 20
