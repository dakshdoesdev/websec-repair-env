"""CLI entrypoint for orchestrated WebSec runs."""

from __future__ import annotations

import argparse

from websec_repair_env.orchestration.controller import default_run_config, run_once, run_tournament, run_watch_loop
from websec_repair_env.orchestration.reporting import render_terminal_summary
from websec_repair_env.orchestration.types import RunConfig


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_config(args: argparse.Namespace) -> RunConfig:
    defaults = default_run_config()
    return RunConfig(
        mode=args.mode,
        base_url=args.base_url,
        task_ids=_split_csv(args.tasks),
        episodes=args.episodes,
        max_steps=args.max_steps,
        retry_limit=args.retry_limit,
        advance_threshold=args.advance_threshold,
        policy_name=args.policy,
        model_name=args.model_name,
        api_base_url=args.api_base_url,
        artifacts_dir=args.artifacts_dir,
        watch=args.watch,
        watch_interval=args.watch_interval,
        models=_split_csv(args.models),
    )


def main() -> None:
    defaults = default_run_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "benchmark", "curriculum", "tournament"], default=defaults.mode)
    parser.add_argument("--base-url", default=defaults.base_url)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--episodes", type=int, default=defaults.episodes)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--retry-limit", type=int, default=defaults.retry_limit)
    parser.add_argument("--advance-threshold", type=float, default=defaults.advance_threshold)
    parser.add_argument("--policy", choices=["local-llm", "heuristic"], default=defaults.policy_name)
    parser.add_argument("--model-name", default=defaults.model_name)
    parser.add_argument("--api-base-url", default=defaults.api_base_url)
    parser.add_argument("--artifacts-dir", default=defaults.artifacts_dir)
    parser.add_argument("--models", default="")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-interval", type=int, default=defaults.watch_interval)
    args = parser.parse_args()

    config = build_config(args)
    if config.watch:
        run_watch_loop(config)
        return
    if config.mode == "tournament":
        for summary in run_tournament(config):
            print(render_terminal_summary(summary))
        return
    summary = run_once(config)
    print(render_terminal_summary(summary))


if __name__ == "__main__":
    main()
