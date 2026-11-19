from websec_repair_env.live_dashboard import render_score_bar, render_workflow_line
from websec_repair_env.terminal_ui import BaselineTui, baseline_plan, next_view_index


def test_render_score_bar_clamps_and_fills() -> None:
    assert render_score_bar(0.5, width=10) == "█████·····"
    assert render_score_bar(2.0, width=5) == "█████"
    assert render_score_bar(-1.0, width=5) == "·····"


def test_render_workflow_line_marks_current_stage() -> None:
    text = render_workflow_line("patched").plain
    assert "● Load" in text
    assert "● Patch" in text
    assert "○ Verify" in text


def test_next_view_index_cycles_both_directions() -> None:
    assert next_view_index(0) == 1
    assert next_view_index(0, reverse=True) == 2


def test_baseline_plan_has_expected_submit_tail() -> None:
    plan = baseline_plan("sqli_login")
    assert plan[0].action_type == "inspect"
    assert plan[-1].action_type == "submit"


def test_task_panel_renders_without_name_error() -> None:
    tui = BaselineTui(base_url="http://127.0.0.1:8000", task_id="sqli_login", model_name="qwen2.5:1.5b")
    panel = tui._task_panel("sqli_login")
    assert panel is not None
