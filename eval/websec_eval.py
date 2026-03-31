#!/usr/bin/env python3
"""Evaluate a fine-tuned WebSec repair model on a fixed defensive task set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.inference import (  # noqa: E402
    WebSecRepairAgent,
    extract_env_action,
    extract_json_object,
    parse_model_output,
)


def load_eval_cases(filepath: Path | str) -> list[dict[str, Any]]:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Evaluation cases not found: {filepath}")

    cases: list[dict[str, Any]] = []
    with filepath.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "expected" not in record:
                raise ValueError(f"Evaluation case on line {line_number} is missing 'expected'")
            cases.append(record)
    return cases


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    action_correct = 0
    vuln_correct = 0
    patch_correct = 0
    vuln_total = 0
    patch_total = 0
    dual_json_valid = 0
    env_action_valid = 0

    for result in results:
        expected = result["expected"]
        predicted = result.get("predicted_action", {})
        if result.get("valid_dual_json", False):
            dual_json_valid += 1
        if result.get("valid_env_action", False):
            env_action_valid += 1

        if predicted.get("action_type") == expected.get("action_type"):
            action_correct += 1

        expected_vuln = expected.get("vulnerability_type")
        if expected_vuln is not None:
            vuln_total += 1
            if predicted.get("vulnerability_type") == expected_vuln:
                vuln_correct += 1

        expected_patch = expected.get("patch_id")
        if expected_patch is not None:
            patch_total += 1
            if predicted.get("patch_id") == expected_patch:
                patch_correct += 1

    return {
        "total_cases": total,
        "action_type_accuracy": action_correct / total if total else 0.0,
        "vulnerability_type_accuracy": vuln_correct / vuln_total if vuln_total else 0.0,
        "patch_id_accuracy": patch_correct / patch_total if patch_total else 0.0,
        "valid_dual_json_rate": dual_json_valid / total if total else 0.0,
        "valid_env_action_extraction_rate": env_action_valid / total if total else 0.0,
        "evaluated_vulnerability_cases": vuln_total,
        "evaluated_patch_cases": patch_total,
    }


def evaluate_model(model_dir: Path | str, cases_path: Path | str) -> dict[str, Any]:
    cases = load_eval_cases(cases_path)
    agent = WebSecRepairAgent(model_dir)

    results: list[dict[str, Any]] = []
    try:
        for case in cases:
            raw_text = agent.generate_raw_response(
                task_id=case["task_id"],
                instruction=case["instruction"],
                code_snippet=case["input"]["code_snippet"],
                scanner_hint=case["input"]["scanner_hint"],
            )

            valid_dual_json = False
            valid_env_action = False
            predicted_action: dict[str, Any] = {"action_type": "inspect"}

            try:
                raw_payload = extract_json_object(raw_text)
                valid_dual_json = isinstance(raw_payload.get("action"), dict)
            except ValueError:
                raw_payload = None

            try:
                parsed = parse_model_output(raw_text)
                predicted_action = extract_env_action(parsed)
                valid_env_action = True
            except ValueError:
                parsed = None

            results.append(
                {
                    "task_id": case["task_id"],
                    "category": case.get("category", "unknown"),
                    "expected": case["expected"],
                    "predicted_action": predicted_action,
                    "valid_dual_json": valid_dual_json,
                    "valid_env_action": valid_env_action,
                    "raw_payload": raw_payload,
                    "parsed": parsed,
                }
            )
    finally:
        agent.unload()

    return {
        "metrics": compute_metrics(results),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned WebSec repair model")
    parser.add_argument("--model-dir", required=True, help="Directory containing adapter and tokenizer")
    parser.add_argument(
        "--cases",
        default=str(REPO_ROOT / "eval" / "websec_eval_cases.jsonl"),
        help="Path to the evaluation JSONL file",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the full evaluation report as JSON",
    )
    args = parser.parse_args()

    report = evaluate_model(args.model_dir, args.cases)
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nSaved detailed report to {output_path}")


if __name__ == "__main__":
    main()

