#!/usr/bin/env python3
"""Inference helpers for the fine-tuned WebSec repair model.

The model is trained to emit a dual JSON payload:

{
  "action": {...},
  "review": {...}
}

For backward compatibility, this module also accepts legacy flat JSON and
exposes flattened `action_type` / `vulnerability_type` / `patch_id` keys in the
returned dictionary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ACTION_TYPES = {"inspect", "classify", "apply_patch", "verify", "submit"}


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the last parseable JSON object from model output."""

    decoder = json.JSONDecoder()
    candidates: list[tuple[int, int, dict[str, Any]]] = []

    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            candidate, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            candidates.append((index, index + end, candidate))

    if not candidates:
        raise ValueError("Model response did not contain a valid JSON object")

    action_candidates = [
        (start, end, candidate)
        for start, end, candidate in candidates
        if "action" in candidate or "action_type" in candidate
    ]
    if action_candidates:
        return max(action_candidates, key=lambda item: (item[1], item[0]))[2]

    return max(candidates, key=lambda item: (item[1], item[0]))[2]


def _clean_review(review: Any) -> dict[str, Any]:
    if not isinstance(review, dict):
        return {}

    cleaned: dict[str, Any] = {}
    for key in ("owasp_top10_2021", "cwe_id", "fix_family", "rationale"):
        value = review.get(key)
        if isinstance(value, str) and value.strip():
            cleaned[key] = value.strip()

    verification_plan = review.get("verification_plan")
    if isinstance(verification_plan, list):
        plan = [str(item).strip() for item in verification_plan if str(item).strip()]
        if plan:
            cleaned["verification_plan"] = plan

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


def normalize_action_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize dual or legacy output into one canonical dictionary."""

    if "action" in payload and isinstance(payload["action"], dict):
        action_source = payload["action"]
        review_source = payload.get("review")
    else:
        action_source = payload
        review_source = payload.get("review")

    action_type = action_source.get("action_type", "inspect")
    if action_type not in ACTION_TYPES:
        action_type = "inspect"

    action: dict[str, Any] = {"action_type": action_type}
    vulnerability_type = action_source.get("vulnerability_type")
    if isinstance(vulnerability_type, str) and vulnerability_type.strip():
        action["vulnerability_type"] = vulnerability_type.strip()

    patch_id = action_source.get("patch_id")
    if isinstance(patch_id, str) and patch_id.strip():
        action["patch_id"] = patch_id.strip()

    normalized: dict[str, Any] = {
        "action": action,
        "action_type": action["action_type"],
    }
    if "vulnerability_type" in action:
        normalized["vulnerability_type"] = action["vulnerability_type"]
    if "patch_id" in action:
        normalized["patch_id"] = action["patch_id"]

    cleaned_review = _clean_review(review_source)
    if cleaned_review:
        normalized["review"] = cleaned_review

    return normalized


def parse_model_output(text: str) -> dict[str, Any]:
    """Parse raw model text into the canonical dual/flat compatibility shape."""

    return normalize_action_payload(extract_json_object(text))


def extract_env_action(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract only the environment-facing action block."""

    normalized = normalize_action_payload(payload)
    return dict(normalized["action"])


class WebSecRepairAgent:
    """Agent that uses a fine-tuned Qwen2.5-Coder model for vulnerability repair."""

    def __init__(self, model_dir: str | Path, device: str = "auto"):
        self.model_dir = Path(model_dir)
        self.device = device

        print(f"Loading model from {self.model_dir}...")

        try:
            import torch
            from peft import AutoPeftModelForCausalLM
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised in runtime use
            raise ImportError(
                "Training dependencies are required for model inference. "
                "Install training_requirements.txt or the [train] extra."
            ) from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir / "tokenizer",
            trust_remote_code=True,
            local_files_only=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoPeftModelForCausalLM.from_pretrained(
            self.model_dir / "adapter",
            device_map=device,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

        print(f"✓ Model loaded on {device}")

    def _build_prompt(
        self,
        task_id: str,
        instruction: str,
        code_snippet: str,
        scanner_hint: str,
    ) -> str:
        return f"""You are an AI assistant helping to fix web security vulnerabilities.

Task: {task_id}
Instruction: {instruction}

Code to review:
```
{code_snippet}
```

Scanner hint: {scanner_hint}

Rules:
- If the code and hint clearly identify a known issue, use action.action_type=\"classify\" with a granular vulnerability_type such as sql_injection, xss, broken_auth, insecure_authentication, path_traversal, file_inclusion, command_injection, or broken_access_control.
- Use inspect only when the evidence is genuinely insufficient.
- If safe patch options are implied by the task, prefer precise patch families like parameterized_query, html_escape, require_admin_role, validate_and_normalize_path, remove_shell_true, or remove_hardcoded_secret.
- Respond with JSON only.

Example 1:
{{
  "action": {{
    "action_type": "classify",
    "vulnerability_type": "sql_injection"
  }},
  "review": {{
    "owasp_top10_2021": "A03:2021-Injection",
    "cwe_id": "CWE-89",
    "rationale": "User-controlled input reaches a SQL query string directly."
  }}
}}

Example 2:
{{
  "action": {{
    "action_type": "classify",
    "vulnerability_type": "path_traversal"
  }},
  "review": {{
    "owasp_top10_2021": "A01:2021-Broken Access Control",
    "cwe_id": "CWE-22",
    "rationale": "User input controls a filesystem path without normalization or allowlisting."
  }}
}}
"""

    def generate_raw_response(
        self,
        task_id: str,
        instruction: str,
        code_snippet: str,
        scanner_hint: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:
        prompt = self._build_prompt(task_id, instruction, code_snippet, scanner_hint)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=2048,
            truncation=True,
        ).to(self.model.device)

        with self._torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_action(
        self,
        task_id: str,
        instruction: str,
        code_snippet: str,
        scanner_hint: str,
    ) -> dict[str, Any]:
        """Get the model's structured recommendation.

        Returns the full dual JSON when available, plus flattened top-level action
        fields for backward compatibility with existing scripts.
        """
        response = self.generate_raw_response(
            task_id,
            instruction,
            code_snippet,
            scanner_hint,
        )
        try:
            return parse_model_output(response)
        except ValueError:
            return normalize_action_payload({"action_type": "inspect"})

    def get_env_action(
        self,
        task_id: str,
        instruction: str,
        code_snippet: str,
        scanner_hint: str,
    ) -> dict[str, Any]:
        """Return only the current environment-compatible action block."""

        return extract_env_action(
            self.get_action(task_id, instruction, code_snippet, scanner_hint)
        )

    def unload(self) -> None:
        """Free model memory."""

        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if hasattr(self, "_torch"):
            self._torch.cuda.empty_cache()


def main() -> None:
    """Demo usage."""

    import argparse

    parser = argparse.ArgumentParser(description="Infer with fine-tuned WebSec model")
    parser.add_argument(
        "model_dir",
        help="Directory containing trained adapter and tokenizer",
    )
    parser.add_argument(
        "--task-id",
        default="sqli_login",
        help="Task ID to test",
    )
    parser.add_argument(
        "--instruction",
        default="Inspect the code and classify the vulnerability type.",
        help="Instruction for the model",
    )
    parser.add_argument(
        "--code-snippet",
        default='query = f"SELECT * FROM users WHERE id={user_id}"',
        help="Inline code snippet to review",
    )
    parser.add_argument(
        "--code-file",
        default=None,
        help="Optional file containing the code snippet to review",
    )
    parser.add_argument(
        "--scanner-hint",
        default="SQL injection detected in query construction",
        help="Scanner hint describing the issue",
    )
    parser.add_argument(
        "--env-action-only",
        action="store_true",
        help="Print only the environment-facing action block",
    )

    args = parser.parse_args()

    agent = WebSecRepairAgent(args.model_dir)
    code_snippet = args.code_snippet
    if args.code_file:
        code_snippet = Path(args.code_file).read_text(encoding="utf-8")

    response = agent.get_action(
        task_id=args.task_id,
        instruction=args.instruction,
        code_snippet=code_snippet,
        scanner_hint=args.scanner_hint,
    )

    print("\nModel recommendation:")
    if args.env_action_only:
        print(json.dumps(extract_env_action(response), indent=2))
    else:
        print(json.dumps(response, indent=2))

    agent.unload()


if __name__ == "__main__":
    main()
