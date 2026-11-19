"""Persistent Rich terminal UI for the WebSec Repair baseline."""

from __future__ import annotations

import select
import sys
import termios
import time
import tty
from contextlib import contextmanager
from dataclasses import dataclass, field

from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import ProgressBar
from rich.table import Table
from rich.text import Text

from .baseline_policy import LocalLLMPolicy, TASK_TO_PATCH, TASK_TO_VULNERABILITY, build_policy_config
from .client import WebSecRepairEnv
from .models import WebSecRepairAction, WebSecRepairObservation

TASK_ORDER = ["sqli_login", "xss_comments", "broken_auth_admin"]
TASK_TITLES = {
    "sqli_login": "EASY - SQLI LOGIN",
    "xss_comments": "MEDIUM - XSS COMMENTS",
    "broken_auth_admin": "HARD - BROKEN AUTH ADMIN",
}
TASK_DIFFICULTY = {
    "sqli_login": "easy",
    "xss_comments": "medium",
    "broken_auth_admin": "hard",
}
VIEWS = ["DASHBOARD", "EVIDENCE", "HISTORY"]
SPARKS = " ▁▂▃▄▅▆▇█"


def next_view_index(current: int, *, reverse: bool = False) -> int:
    if reverse:
        return (current - 1) % len(VIEWS)
    return (current + 1) % len(VIEWS)


def baseline_plan(task_id: str) -> list[WebSecRepairAction]:
    return [
        WebSecRepairAction(action_type="inspect"),
        WebSecRepairAction(
            action_type="classify",
            vulnerability_type=TASK_TO_VULNERABILITY[task_id],
        ),
        WebSecRepairAction(
            action_type="apply_patch",
            patch_id=TASK_TO_PATCH[task_id],
        ),
        WebSecRepairAction(action_type="verify"),
        WebSecRepairAction(action_type="submit"),
    ]


def _sparkline(values: list[float], width: int = 20) -> str:
    if not values:
        return "[grey35]waiting...[/grey35]"
    tail = values[-width:]
    lo = min(tail)
    hi = max(tail)
    if hi == lo:
        color = "green" if hi >= 0 else "red"
        return f"[{color}]" + ("█" * len(tail)) + f"[/{color}]"

    chars: list[str] = []
    for value in tail:
        ratio = (value - lo) / (hi - lo)
        idx = min(len(SPARKS) - 1, max(0, int(round(ratio * (len(SPARKS) - 1)))))
        chars.append(SPARKS[idx])
    color = "green" if tail[-1] >= 0 else "red"
    return f"[{color}]{''.join(chars)}[/{color}]"


def _action_label(action_type: str) -> tuple[str, str]:
    mapping = {
        "inspect": ("INV", "cyan"),
        "classify": ("CLS", "yellow"),
        "apply_patch": ("FIX", "green"),
        "verify": ("VFY", "magenta"),
        "submit": ("END", "bright_green"),
    }
    return mapping.get(action_type, ("ACT", "white"))


def _status_style(status: str) -> str:
    return {
        "RUNNING": "bold bright_yellow",
        "READY": "grey62",
        "SUCCESS": "bold bright_green",
        "FAILED": "bold bright_red",
        "PAUSED": "bold bright_cyan",
        "BOOT": "grey50",
    }.get(status, "white")


def _border_style(status: str, active: bool = False) -> str:
    if active and status == "RUNNING":
        return "bright_yellow"
    if active:
        return "bright_cyan"
    if status == "SUCCESS":
        return "bright_green"
    if status == "FAILED":
        return "bright_red"
    if status == "RUNNING":
        return "yellow"
    return "grey35"


@contextmanager
def _raw_input_mode():
    if not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_key() -> str | None:
    if not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return None
    first = sys.stdin.read(1)
    if first != "\x1b":
        return first
    ready, _, _ = select.select([sys.stdin], [], [], 0.001)
    if not ready:
        return first
    second = sys.stdin.read(1)
    if second != "[":
        return first + second
    third = sys.stdin.read(1)
    if third == "Z":
        return "shift-tab"
    if third == "C":
        return "right"
    if third == "D":
        return "left"
    return first + second + third


@dataclass
class TaskSummary:
    status: str = "READY"
    steps: int = 0
    score: float = 0.0
    total_reward: float = 0.0
    history: list[float] = field(default_factory=list)


@dataclass
class EventEntry:
    step: int
    action_type: str
    reward: float
    status_message: str
    source: str


@dataclass
class TuiState:
    task_id: str
    model_name: str
    start_time: float = field(default_factory=time.time)
    observation: WebSecRepairObservation | None = None
    total_reward: float = 0.0
    last_reward: float = 0.0
    step_count: int = 0
    score: float = 0.0
    status: str = "BOOT"
    done: bool = False
    success: bool = False
    auto_running: bool = True
    next_auto_at: float = 0.0
    view_index: int = 0
    reward_history: list[float] = field(default_factory=list)
    cumulative_history: list[float] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    history: list[dict[str, object]] = field(default_factory=list)
    events: list[EventEntry] = field(default_factory=list)
    task_runs: dict[str, TaskSummary] = field(
        default_factory=lambda: {task_id: TaskSummary() for task_id in TASK_ORDER}
    )

    @property
    def view(self) -> str:
        return VIEWS[self.view_index]


class BaselineTui:
    def __init__(self, *, base_url: str, task_id: str, model_name: str) -> None:
        self.base_url = base_url
        self.console = Console()
        self.state = TuiState(task_id=task_id, model_name=model_name)
        self.policy = LocalLLMPolicy(build_policy_config(model_name=model_name))
        self.env: WebSecRepairEnv | None = None

    def run(self) -> int:
        self.env = self._connect_with_retry()
        try:
            self._reset_episode(auto_start=True)
            with _raw_input_mode():
                with Live(self._build_layout(), console=self.console, refresh_per_second=8, screen=True) as live:
                    while True:
                        self._tick_auto()
                        key = _read_key()
                        if key and not self._handle_key(key):
                            return 0
                        live.update(self._build_layout(), refresh=True)
                        time.sleep(0.05)
        except KeyboardInterrupt:
            return 0
        finally:
            if self.env is not None:
                self.env.__exit__(None, None, None)

    def _connect_with_retry(self) -> WebSecRepairEnv:
        attempts = 15
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                env = WebSecRepairEnv(base_url=self.base_url).sync()
                env.__enter__()
                return env
            except Exception as exc:  # pragma: no cover - runtime guard
                last_error = exc
                time.sleep(0.4)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to connect to environment")

    def _reset_episode(self, *, auto_start: bool) -> None:
        assert self.env is not None
        result = self.env.reset(task_id=self.state.task_id)
        self.state.observation = result.observation
        self.state.total_reward = 0.0
        self.state.last_reward = 0.0
        self.state.step_count = 0
        self.state.score = 0.0
        self.state.status = "RUNNING" if auto_start else "READY"
        self.state.done = False
        self.state.success = False
        self.state.auto_running = auto_start
        self.state.next_auto_at = time.time() + 0.35
        self.state.reward_history = []
        self.state.cumulative_history = []
        self.state.logs = [f"reset {self.state.task_id} on {self.base_url}"]
        self.state.history = []
        self.state.events = []
        self.state.task_runs[self.state.task_id] = TaskSummary(status="RUNNING")

    def _record_step(self, action: WebSecRepairAction, source: str, result, reward: float) -> None:
        self.state.observation = result.observation
        self.state.last_reward = reward
        self.state.total_reward += reward
        self.state.step_count += 1
        self.state.score = self.state.total_reward
        self.state.done = bool(result.done)
        self.state.success = bool(result.observation.grader_passed and result.done)
        self.state.status = "SUCCESS" if self.state.success else "FAILED" if self.state.done else "RUNNING"
        self.state.reward_history.append(reward)
        self.state.cumulative_history.append(self.state.total_reward)
        self.state.history.append(
            {
                "step": self.state.step_count,
                "action": action.action_type,
                "reward": reward,
                "status_message": result.observation.status_message,
            }
        )
        self.state.events.append(
            EventEntry(
                step=self.state.step_count,
                action_type=action.action_type,
                reward=reward,
                status_message=result.observation.status_message,
                source=source,
            )
        )
        suffix = f" [{source}]" if source != "local-llm" else ""
        self.state.logs.append(
            f"t={self.state.step_count:02d} [F] {action.action_type:<10} {reward:+.2f}{suffix}"
        )
        summary = self.state.task_runs[self.state.task_id]
        summary.status = self.state.status
        summary.steps = self.state.step_count
        summary.score = self.state.score
        summary.total_reward = self.state.total_reward
        summary.history = list(self.state.cumulative_history)

    def _tick_auto(self) -> None:
        if not self.state.auto_running or self.state.done:
            return
        if time.time() < self.state.next_auto_at:
            return
        if self.state.observation is None:
            return

        action, source, _ = self.policy.choose_action(self.state.observation, self.state.history)
        assert self.env is not None
        result = self.env.step(action)
        reward = float(result.reward or 0.0)
        self._record_step(action, source, result, reward)
        self.state.next_auto_at = time.time() + 0.9

    def _handle_key(self, key: str) -> bool:
        if key.lower() == "q":
            return False
        if key == "\t":
            self.state.view_index = next_view_index(self.state.view_index)
        elif key == "shift-tab":
            self.state.view_index = next_view_index(self.state.view_index, reverse=True)
        elif key in {"1", "2", "3"}:
            self.state.task_id = TASK_ORDER[int(key) - 1]
            self._reset_episode(auto_start=True)
        elif key in {"left", "h"}:
            idx = (TASK_ORDER.index(self.state.task_id) - 1) % len(TASK_ORDER)
            self.state.task_id = TASK_ORDER[idx]
            self._reset_episode(auto_start=True)
        elif key in {"right", "l"}:
            idx = (TASK_ORDER.index(self.state.task_id) + 1) % len(TASK_ORDER)
            self.state.task_id = TASK_ORDER[idx]
            self._reset_episode(auto_start=True)
        elif key.lower() == "r":
            self._reset_episode(auto_start=True)
        elif key in {" ", "\n", "\r"}:
            if self.state.done:
                self._reset_episode(auto_start=True)
            else:
                self.state.auto_running = not self.state.auto_running
                self.state.status = "PAUSED" if not self.state.auto_running else "RUNNING"
                self.state.next_auto_at = time.time() + 0.1
        return True

    def _build_layout(self) -> Layout:
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="top", size=9),
            Layout(name="middle", size=12),
            Layout(name="bottom"),
        )
        layout["header"].update(self._header_panel())
        layout["top"].split_row(*(Layout(name=task) for task in TASK_ORDER))
        for task in TASK_ORDER:
            layout["top"][task].update(self._task_panel(task))
        layout["middle"].split_row(Layout(name="agent"), Layout(name="evidence"))
        layout["middle"]["agent"].update(self._agent_panel())
        layout["middle"]["evidence"].update(self._evidence_panel())
        layout["bottom"].split_row(Layout(name="reward"), Layout(name="log"))
        layout["bottom"]["reward"].update(self._reward_panel())
        layout["bottom"]["log"].update(self._log_panel())
        return layout

    def _header_panel(self) -> Panel:
        elapsed = int(time.time() - self.state.start_time)
        status_style = _status_style(self.state.status)
        help_text = Text.assemble(
            ("Tab/Shift-Tab", "bold cyan"),
            (" views  ", "grey62"),
            ("1/2/3", "bold yellow"),
            (" task presets  ", "grey62"),
            ("←/→", "bold yellow"),
            (" switch  ", "grey62"),
            ("space", "bold green"),
            (" pause  ", "grey62"),
            ("r", "bold magenta"),
            (" rerun  ", "grey62"),
            ("q", "bold red"),
            (" quit", "grey62"),
        )
        body = Group(
            Text.assemble(
                ("WebSec Repair Dashboard", "bold bright_white"),
                (" | ", "grey62"),
                (f"model={self.state.model_name}", "bright_cyan"),
                (" | ", "grey62"),
                (f"task={self.state.task_id}", "bright_yellow"),
                (" | ", "grey62"),
                (f"elapsed {elapsed // 60:02d}:{elapsed % 60:02d}", "white"),
                (" | ", "grey62"),
                (self.state.status.lower(), status_style),
            ),
            help_text,
        )
        return Panel(body, style="on #10151d", border_style="bright_cyan", padding=(0, 1))

    def _task_panel(self, task_id: str) -> Panel:
        summary = self.state.task_runs[task_id]
        active = task_id == self.state.task_id
        border = _border_style(summary.status, active)
        score_style = "bold bright_green" if summary.score >= 0.9 else "bold bright_yellow" if summary.score >= 0.4 else "bold bright_red"
        reward_style = "bold bright_green" if summary.total_reward >= 0 else "bold bright_red"
        trend = _sparkline(summary.history, width=18)

        metrics = Table.grid(expand=True)
        metrics.add_column()
        metrics.add_column()
        metrics.add_column()
        metrics.add_row(
            Text(f"{summary.score:+.2f}", style=score_style),
            Text(f"{summary.total_reward:+.2f}", style=reward_style),
            Text(f"{summary.steps}/6", style="bold bright_cyan"),
        )
        metrics.add_row(
            Text("score", style="grey62"),
            Text("reward", style="grey62"),
            Text("steps", style="grey62"),
        )
        metrics.add_row(
            Text(trend, style=reward_style),
            Text(summary.status, style=score_style),
            Text(TASK_DIFFICULTY[task_id], style="bright_cyan"),
        )
        title = Text(f" {TASK_TITLES[task_id]} ", style="bold white")
        return Panel(Align.center(metrics, vertical="middle"), title=title, border_style=border)

    def _agent_panel(self) -> Panel:
        obs = self.state.observation
        reward_style = "bold bright_green" if self.state.total_reward >= 0 else "bold bright_red"
        status_style = _status_style(self.state.status)
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(style="grey62", width=12)
        table.add_column(style="bold white")
        table.add_row("tick", Text(f"{self.state.step_count}/6", style="bold bright_yellow"))
        table.add_row("status", Text(self.state.status, style=status_style))
        table.add_row("reward", Text(f"last {self.state.last_reward:+.2f}   total {self.state.total_reward:+.2f}", style=reward_style))
        table.add_row("", Text(""))

        health = Table.grid(expand=True)
        health.add_column(width=14)
        health.add_column(width=5)
        health.add_column()
        exploit = bool(obs and obs.exploit_test_passed)
        func = bool(obs and obs.functionality_test_passed)
        health.add_row(
            "Exploit Probe",
            Text("YES" if exploit else "NO", style="bold bright_green" if exploit else "bold bright_red"),
            ProgressBar(total=1.0, completed=1.0 if exploit else 0.0, width=12, style="green" if exploit else "red"),
        )
        health.add_row(
            "Func. Tests",
            Text("YES" if func else "NO", style="bold bright_green" if func else "bold bright_red"),
            ProgressBar(total=1.0, completed=1.0 if func else 0.0, width=12, style="green" if func else "red"),
        )

        latest = self.state.events[-1] if self.state.events else None
        latest_text = "Waiting for first action..."
        if latest:
            latest_text = f"{latest.action_type:<10} {latest.reward:+.2f} {latest.source}"

        body = Group(
            table,
            Text("SYSTEM VERIFICATION", style="grey62"),
            health,
            Text(""),
            Panel(Text(f"► {latest_text}", style="bold white"), border_style="bright_green", padding=(0, 1)),
        )
        return Panel(body, title=Text(" REPAIR AGENT ", style="bold white"), border_style="bright_yellow")

    def _evidence_panel(self) -> Panel:
        obs = self.state.observation
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(style="bright_cyan", width=14)
        table.add_column(style="white")
        if obs:
            table.add_row("Workflow:", Text(obs.workflow_stage, style="bold bright_magenta"))
            table.add_row("Scanner:", Text(obs.selected_vulnerability or "-", style="bold bright_yellow"))
            table.add_row("Patch Template:", Text(obs.applied_patch_id or "-", style="bold bright_magenta"))
            hint_style = "bright_green" if "solved" in obs.status_message.lower() or "blocked" in obs.status_message.lower() else "bright_red" if "failed" in obs.status_message.lower() or "requires" in obs.status_message.lower() else "italic"
            table.add_row("Hint:", Text(obs.status_message[:160], style=hint_style))
        else:
            table.add_row("Hint:", Text("Waiting for task reset...", style="grey62"))
        title = Text(f" {self.state.view} ", style="bold white") if self.state.view != "DASHBOARD" else Text(" SCAN EVIDENCE ", style="bold white")
        return Panel(table, title=title, border_style="bright_cyan")

    def _reward_panel(self) -> Panel:
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(style="grey62", width=16)
        table.add_column()
        score_style = "bold bright_green" if self.state.score >= 0.9 else "bold bright_yellow" if self.state.score >= 0.4 else "bold bright_red"
        table.add_row("Cumulative", Text(_sparkline(self.state.cumulative_history, width=30), style=score_style))
        table.add_row("Current Step", Text(_sparkline(self.state.reward_history, width=30), style="bold bright_yellow"))
        table.add_row("", Text(""))
        rewards = [
            ("inspect first", "+0.05"),
            ("correct classify", "+0.20"),
            ("correct patch", "+0.30"),
            ("verify exploit", "+0.15"),
            ("verify functionality", "+0.15"),
            ("verify bonus", "+0.05"),
            ("submit success", "+0.10"),
        ]
        for label, value in rewards:
            table.add_row(label, Text(value, style="bold bright_green"))
        return Panel(table, title=Text(" LIVE REWARD STREAM ", style="bold white"), border_style="bright_green")

    def _log_panel(self) -> Panel:
        if not self.state.events:
            return Panel(Text("Waiting for first event...", style="grey62"), title=Text(" EVENT LOG ", style="bold white"), border_style="bright_magenta")

        lines = []
        for event in self.state.events[-10:]:
            label, label_style = _action_label(event.action_type)
            source_style = "bright_yellow" if "guardrail" in event.source else "bright_green" if event.source == "local-llm" else "bright_red"
            reward_style = "bright_green" if event.reward >= 0 else "bright_red"
            line = Text()
            line.append(f" t={event.step:02d} ", style="grey62")
            line.append("[F] ", style="bright_cyan")
            line.append(f"{label:<3} ", style=label_style)
            line.append(f"{event.action_type:<10}", style="white")
            line.append(f"{event.reward:+.2f} ", style=reward_style)
            line.append(f"[{event.source}]", style=source_style)
            lines.append(line)
        return Panel(Group(*lines), title=Text(" EVENT LOG ", style="bold white"), border_style="bright_magenta")
