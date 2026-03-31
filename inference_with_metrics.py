#!/usr/bin/env python3
"""Deterministic baseline agent for WebSec Repair Env with metrics tracking."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from websec_repair_env import WebSecRepairAction, WebSecRepairEnv
from visualization import MetricsTracker, ComparisonPlotter


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


def run_baseline(base_url: str, task_id: str, tracker: MetricsTracker | None = None) -> int:
    """Run the deterministic baseline policy against a running environment.
    
    Args:
        base_url: Server URL
        task_id: Task ID to run
        tracker: Optional MetricsTracker to record metrics
    
    Returns:
        0 if passed, 1 if failed
    """
    start_time = time.time()
    episode = getattr(tracker, '_episode_num', 1) if tracker else 1
    
    with WebSecRepairEnv(base_url=base_url).sync() as env:
        result = env.reset(task_id=task_id)
        print(f"reset: task={result.observation.task_id}")

        result = env.step(WebSecRepairAction(action_type="inspect"))
        print(f"inspect: reward={result.reward} status={result.observation.status_message}")

        vulnerability = choose_vulnerability(
            result.observation.task_id,
            result.observation.scanner_hint,
        )
        vuln_correct = (vulnerability == result.observation.selected_vulnerability)
        
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
        patch_correct = (patch_id == result.observation.applied_patch_id)
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
        
        # Record metrics if tracker provided
        if tracker:
            elapsed = time.time() - start_time
            tracker.add_episode(
                episode=episode,
                task_id=task_id,
                cumulative_reward=result.reward,
                success=result.observation.grader_passed,
                steps=5,  # Fixed: inspect, classify, apply_patch, verify, submit
                vulnerability_correct=vuln_correct,
                patch_correct=patch_correct,
                time_seconds=elapsed,
            )
        
        return 0 if result.observation.grader_passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline agent with optional metrics tracking")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--task",
        default="sqli_login",
        choices=sorted(TASK_TO_VULNERABILITY),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Enable metrics tracking and visualization"
    )
    parser.add_argument(
        "--output-dir",
        default="baseline_results",
        help="Output directory for metrics and plots"
    )
    args = parser.parse_args()
    
    # Run episodes
    tracker = None
    if args.metrics:
        tracker = MetricsTracker(run_name="baseline_agent")
    
    for episode in range(1, args.episodes + 1):
        print(f"\n{'='*60}")
        print(f"Episode {episode}/{args.episodes}")
        print('='*60)
        
        if tracker:
            tracker._episode_num = episode
        
        result = run_baseline(args.base_url, args.task, tracker)
        
        if result == 0:
            print(f"✅ Episode {episode} PASSED")
        else:
            print(f"❌ Episode {episode} FAILED")
    
    # Generate report if metrics enabled
    if tracker and args.metrics:
        print(f"\n{'='*60}")
        print("📊 Generating Metrics Report")
        print('='*60)
        
        agg = tracker.get_aggregate()
        print(f"Success Rate: {agg.success_pct:.1f}%")
        print(f"Avg Reward: {agg.avg_reward:.3f}")
        print(f"Vulnerability Accuracy: {agg.vulnerability_accuracy*100:.1f}%")
        print(f"Patch Accuracy: {agg.patch_accuracy*100:.1f}%")
        
        # Save metrics
        tracker.save_json(f"{args.output_dir}/metrics.json")
        print(f"✅ Metrics saved to {args.output_dir}/metrics.json")


if __name__ == "__main__":
    main()
