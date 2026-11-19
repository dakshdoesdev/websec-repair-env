"""Rich terminal dashboard for the WebSec Repair baseline runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

WORKFLOW_ORDER = ["start", "inspected", "classified", "patched", "verified", "submitted"]
WORKFLOW_LABELS = {
    "start": "Load",
    "inspected": "Inspect",
    "classified": "Classify",
    "patched": "Patch",
    "verified": "Verify",
    "submitted": "Submit",
}


def render_score_bar(score: float, width: int = 24) -> str:
    clamped = max(0.0, min(1.0, score))
    filled = int(round(clamped * width))
    return "█" * filled + "·" * (width - filled)


def render_workflow_line(stage: str) -> Text:
    current_idx = WORKFLOW_ORDER.index(stage) if stage in WORKFLOW_ORDER else 0
    text = Text()
    for idx, item in enumerate(WORKFLOW_ORDER):
        label = WORKFLOW_LABELS[item]
        style = "green" if idx < current_idx else "cyan" if idx == current_idx else "grey62"
        prefix = "●" if idx <= current_idx else "○"
        text.append(f"{prefix} {label}", style=style)
        if idx < len(WORKFLOW_ORDER) - 1:
            text.append("  ")
    return text


@dataclass
class DashboardState:
    task_id: str = "-"
    title: str = "-"
    difficulty: str = "-"
    status: str = "idle"
    step_count: int = 0
    max_steps: int = 6
    total_reward: float = 0.0
    last_reward: float = 0.0
    workflow_stage: str = "start"
    selected_vulnerability: str = ""
    applied_patch_id: str = ""
    exploit_test_passed: bool = False
    functionality_test_passed: bool = False
    grader_passed: bool = False
    score: float = 0.0
    current_action: str = "-"
    status_message: str = "Waiting for first step..."
    scanner_hint: str = ""
    code_snippet: str = ""
    patch_options: list[dict[str, Any]] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)


def _status_style(status: str) -> str:
    return {
        "idle": "white",
        "running": "cyan",
        "success": "green",
        "failed": "red",
    }.get(status, "white")


def _bool_text(value: bool, yes: str, no: str) -> str:
    return yes if value else no


def _render_summary(state: DashboardState) -> Panel:
    table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    table.add_column("k", style="cyan", ratio=1)
    table.add_column("v", ratio=2)
    table.add_row("status", f"[{_status_style(state.status)}]{state.status}[/{_status_style(state.status)}]")
    table.add_row("task", f"{state.task_id} ({state.difficulty})")
    table.add_row("steps", f"{state.step_count}/{state.max_steps}")
    table.add_row("reward", f"last={state.last_reward:+.2f} total={state.total_reward:+.2f}")
    table.add_row("score", f"[bold]{state.score:.2f}[/bold]")
    table.add_row("classification", state.selected_vulnerability or "-")
    table.add_row("patch", state.applied_patch_id or "-")
    table.add_row("checks", f"{_bool_text(state.exploit_test_passed, 'exploit ok', 'exploit pending')} / {_bool_text(state.functionality_test_passed, 'functionality ok', 'functionality pending')}")
    table.add_row("grader", _bool_text(state.grader_passed, "passed", "pending"))
    return Panel(table, title="Episode Summary", border_style="bright_blue")


def _render_latest(state: DashboardState) -> Panel:
    body = Text.from_markup(
        f"[bold]workflow[/bold] {state.workflow_stage}\n"
        f"[bold]action[/bold] {state.current_action}\n"
        f"[bold]result[/bold] {state.status_message}"
    )
    return Panel(body, title="Latest Step", border_style="magenta")


def _render_code_and_hint(state: DashboardState) -> Panel:
    snippet = state.code_snippet or "(hidden until inspect)"
    hint = state.scanner_hint or "(hidden until inspect)"
    body = Text()
    body.append("Scanner hint\n", style="bold cyan")
    body.append(hint)
    body.append("\n\nCode snippet\n", style="bold cyan")
    body.append(snippet)
    return Panel(body, title="Evidence", border_style="green")


def _render_patch_options(state: DashboardState) -> Panel:
    table = Table(show_header=True, expand=True)
    table.add_column("Patch ID", style="cyan")
    table.add_column("Label")
    if not state.patch_options:
        table.add_row("-", "Inspect the task to reveal patch options.")
    else:
        for option in state.patch_options:
            label = option.get("label", "")
            patch_id = option.get("id", "")
            if patch_id == state.applied_patch_id:
                label = f"{label} [selected]"
            table.add_row(patch_id, label)
    return Panel(table, title="Patch Options", border_style="yellow")


def _render_progress(state: DashboardState) -> Panel:
    score_bar = render_score_bar(state.score)
    body = Group(
        render_workflow_line(state.workflow_stage),
        Text.assemble(
            ("score ", "bold white"),
            (f"[{score_bar}] ", "green" if state.score >= 0.9 else "yellow" if state.score >= 0.5 else "red"),
            (f"{int(round(max(0.0, min(1.0, state.score)) * 100))}%", "bold white"),
        ),
    )
    return Panel(body, title="Workflow", border_style="cyan")


def _render_log(state: DashboardState) -> Panel:
    lines = "\n".join(state.logs[-8:]) if state.logs else "Waiting for first event..."
    return Panel(lines, title="Event Log", border_style="white")


def _reward_table() -> Table:
    table = Table(show_header=True, box=None, expand=True, padding=(0, 1))
    table.add_column("Reward Event", style="bold")
    table.add_column("Value", justify="right")
    rows = [
        ("First inspect", "+0.05"),
        ("Correct classify", "+0.20"),
        ("Correct patch", "+0.30"),
        ("Verify exploit", "+0.15"),
        ("Verify functionality", "+0.15"),
        ("Verify bonus", "+0.05"),
        ("Submit success", "+0.10"),
        ("Wrong stage / bad move", "penalty"),
    ]
    for name, value in rows:
        table.add_row(name, value)
    return table


def _render_state_checks(state: DashboardState) -> Panel:
    body = Text.from_markup(
        f"[bold]classification[/bold] {state.selected_vulnerability or '-'}\n"
        f"[bold]patch[/bold] {state.applied_patch_id or '-'}\n"
        f"[bold]exploit[/bold] {'PASS' if state.exploit_test_passed else 'pending'}\n"
        f"[bold]functionality[/bold] {'PASS' if state.functionality_test_passed else 'pending'}\n"
        f"[bold]grader[/bold] {'passed' if state.grader_passed else 'pending'}"
    )
    return Panel(body, title="Checks", border_style="green")


def build_layout(state: DashboardState, console: Console) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="top", size=9),
        Layout(name="middle", size=12),
        Layout(name="bottom"),
    )
    layout["top"].split_row(Layout(name="progress"), Layout(name="summary"))
    layout["middle"].split_row(Layout(name="latest"), Layout(name="checks"), Layout(name="reward"))
    layout["bottom"].split_row(Layout(name="patches"), Layout(name="evidence"), Layout(name="log"))

    layout["header"].update(
        Panel(
            Text.assemble(
                (state.title or "WebSec Repair", "bold bright_white"),
                ("\n", ""),
                ("WebSec Repair baseline dashboard", "cyan"),
                (" | ", "dim white"),
                (f"task={state.task_id}", "white"),
                (" | ", "dim white"),
                (f"status={state.status}", _status_style(state.status)),
            ),
            border_style="bright_cyan",
        )
    )
    layout["progress"].update(_render_progress(state))
    layout["summary"].update(_render_summary(state))
    layout["latest"].update(_render_latest(state))
    layout["checks"].update(_render_state_checks(state))
    layout["reward"].update(Panel(_reward_table(), title="Reward System", border_style="yellow"))
    layout["patches"].update(_render_patch_options(state))
    layout["evidence"].update(_render_code_and_hint(state))
    layout["log"].update(_render_log(state))
    return layout


class BaselineDashboard:
    def __init__(self, model_name: str, task_id: str) -> None:
        self.console = Console()
        self.state = DashboardState(task_id=task_id)
        self.model_name = model_name
        self.live: Live | None = None

    def __enter__(self) -> "BaselineDashboard":
        self.live = Live(
            build_layout(self.state, self.console),
            console=self.console,
            refresh_per_second=8,
            screen=True,
        )
        self.live.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.live is not None:
            self.live.__exit__(exc_type, exc, tb)

    def start_episode(self, *, title: str, difficulty: str, max_steps: int) -> None:
        self.state.title = title
        self.state.difficulty = difficulty
        self.state.max_steps = max_steps
        self.state.status = "running"
        self.state.logs.append(f"started {self.state.task_id} with model={self.model_name}")
        self.refresh()

    def record_step(
        self,
        *,
        step_count: int,
        action: str,
        observation: dict[str, Any],
        reward: float,
        total_reward: float,
    ) -> None:
        self.state.step_count = step_count
        self.state.last_reward = reward
        self.state.total_reward = total_reward
        self.state.workflow_stage = observation.get("workflow_stage", self.state.workflow_stage)
        self.state.selected_vulnerability = observation.get("selected_vulnerability", "")
        self.state.applied_patch_id = observation.get("applied_patch_id", "")
        self.state.exploit_test_passed = bool(observation.get("exploit_test_passed", False))
        self.state.functionality_test_passed = bool(observation.get("functionality_test_passed", False))
        self.state.grader_passed = bool(observation.get("grader_passed", False))
        self.state.score = total_reward
        self.state.current_action = action
        self.state.status_message = observation.get("status_message", self.state.status_message)
        self.state.scanner_hint = observation.get("scanner_hint", "")
        self.state.code_snippet = observation.get("code_snippet", "")
        self.state.patch_options = observation.get("patch_options", [])
        self.state.logs.append(
            f"step {self.state.step_count}: reward={reward:+.2f} action={action} -> {self.state.status_message}"
        )
        self.refresh()

    def finish_episode(self, *, success: bool, score: float) -> None:
        self.state.status = "success" if success else "failed"
        self.state.grader_passed = success
        self.state.score = score
        self.state.logs.append(f"episode complete score={score:.2f} success={success}")
        self.refresh()

    def refresh(self) -> None:
        if self.live is not None:
            self.live.update(build_layout(self.state, self.console), refresh=True)
