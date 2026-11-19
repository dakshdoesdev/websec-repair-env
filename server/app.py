"""FastAPI application for the WebSec Repair environment."""

from __future__ import annotations

import argparse

from fastapi import HTTPException
from fastapi.responses import HTMLResponse 

try:
    from openenv.core.env_server.http_server import create_app

    from ..models import BaselineCatalog, GraderReport, RuntimeStatus, TaskCatalog, WebSecRepairAction, WebSecRepairObservation
    from .challenge import current_runtime_progress, grade_task, list_baselines, list_tasks
    from .websec_repair_environment import WebSecRepairEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app

    from models import (  # type: ignore
        BaselineCatalog,
        GraderReport,
        RuntimeStatus,
        TaskCatalog,
        WebSecRepairAction,
        WebSecRepairObservation,
    )
    from server.challenge import current_runtime_progress, grade_task, list_baselines, list_tasks  # type: ignore
    from server.websec_repair_environment import WebSecRepairEnvironment  # type: ignore


app = create_app(
    WebSecRepairEnvironment,
    WebSecRepairAction,
    WebSecRepairObservation,
    env_name="websec_repair_env",
    max_concurrent_envs=1,
)


@app.get("/", response_class=HTMLResponse)
def web_ui():
    """Interactive web UI for testing the environment."""
    return HTMLResponse(content=_build_ui())


@app.get("/tasks", response_model=TaskCatalog, tags=["challenge"])
def tasks(difficulty: str | None = None) -> TaskCatalog:
    """List all deterministic vulnerability repair tasks, optionally by difficulty."""
    try:
        return list_tasks(difficulty=difficulty)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/grader", response_model=GraderReport, tags=["challenge"])
def grader(task_id: str | None = None) -> GraderReport:
    """Return the current grader state, optionally filtered to a task id."""
    try:
        return grade_task(task_id=task_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/baseline", response_model=BaselineCatalog, tags=["challenge"])
def baseline(task_id: str | None = None) -> BaselineCatalog:
    """Return all baseline trajectories or one filtered task baseline."""
    try:
        return list_baselines(task_id=task_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/status", response_model=RuntimeStatus, tags=["challenge"])
def status() -> RuntimeStatus:
    """Return the current in-memory runtime progress and grader outcome."""
    progress = current_runtime_progress()
    report = grade_task(progress=progress)
    return RuntimeStatus(environment="websec_repair_env", progress=progress, grader=report)


def _build_ui() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Web Security Repair — OpenEnv</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }
  .header { background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%); padding: 24px 32px; border-bottom: 2px solid #2563eb; }
  .header h1 { font-size: 1.8rem; font-weight: 700; color: #60a5fa; }
  .header p { color: #94a3b8; margin-top: 4px; font-size: 0.95rem; }
  .main { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 20px 32px; max-width: 1400px; margin: 0 auto; }
  .panel { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; }
  .panel h2 { font-size: 1rem; font-weight: 600; color: #93c5fd; margin-bottom: 14px; display: flex; align-items: center; gap: 8px; }
  .task-btns { display: flex; gap: 10px; margin-bottom: 16px; flex-wrap: wrap; }
  .btn { padding: 10px 20px; border: none; border-radius: 8px; cursor: pointer; font-size: 0.875rem; font-weight: 600; transition: all 0.2s; }
  .btn:disabled { background: #334155; cursor: not-allowed; }
  .btn-primary { background: #2563eb; color: white; }
  .btn-primary:hover:not(:disabled) { background: #1d4ed8; }
  .btn-success { background: #059669; color: white; }
  .btn-success:hover:not(:disabled) { background: #047857; }
  .btn-danger { background: #dc2626; color: white; }
  .btn-danger:hover:not(:disabled) { background: #b91c1c; }
  .btn-sm { padding: 6px 12px; font-size: 0.8rem; }
  textarea, select, input { width: 100%; background: #0f172a; border: 1px solid #475569; border-radius: 8px; color: #e2e8f0; padding: 10px; font-size: 0.875rem; font-family: inherit; resize: vertical; }
  textarea:focus, select:focus, input:focus { outline: none; border-color: #2563eb; }
  .output { background: #0f172a; border: 1px solid #1e3a5f; border-radius: 8px; padding: 14px; font-family: 'Courier New', monospace; font-size: 0.8rem; min-height: 100px; max-height: 420px; overflow-y: auto; white-space: pre-wrap; line-height: 1.5; color: #a5f3fc; }
  .score-bar { margin-top: 12px; }
  .score-track { background: #0f172a; border-radius: 9999px; height: 20px; overflow: hidden; border: 1px solid #334155; }
  .score-fill { height: 100%; border-radius: 9999px; transition: width 0.6s ease; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 700; color: white; }
  .action-row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px; }
  label { display: block; font-size: 0.8rem; color: #94a3b8; margin-bottom: 4px; }
  .full { grid-column: 1 / -1; }
  .status-row { display: flex; gap: 16px; margin-bottom: 16px; }
  .stat { text-align: center; }
  .stat-label { font-size: 0.7rem; color: #64748b; }
  .step-count { font-size: 1.5rem; font-weight: 700; color: #fbbf24; }
</style>
</head>
<body>
<div class="header">
  <h1>🛡️ Web Security Repair</h1>
  <p>OpenEnv RL Environment — Train AI agents to fix real security vulnerabilities</p>
</div>

<div class="main">
  <!-- Left: Control Panel -->
  <div>
    <div class="panel" style="margin-bottom:16px">
      <h2>Select Task</h2>
      <div id="task-buttons" class="task-btns"></div>
    </div>

    <div class="panel" style="margin-bottom:16px">
      <h2>Status</h2>
      <div class="status-row">
        <div class="stat">
          <div class="step-count" id="step-count">—</div>
          <div class="stat-label">Steps Taken</div>
        </div>
      </div>
      <div id="observation-display"></div>
    </div>

    <div class="panel">
      <h2>Take Action</h2>
      <div class="action-row">
        <div>
          <label>Action</label>
          <select id="action-select">
            <option value="inspect">inspect</option>
            <option value="classify">classify</option>
            <option value="apply_patch">apply_patch</option>
            <option value="verify">verify</option>
            <option value="submit">submit</option>
          </select>
        </div>
        <div>
          <label id="args-label">Arguments</label>
          <input type="text" id="action-args" placeholder='e.g., {"vulnerability_type": "xss"}'>
        </div>
      </div>
      <button id="execute-btn" class="btn btn-primary" onclick="takeAction()" disabled>Execute Action</button>
      <button id="grade-btn" class="btn btn-success btn-sm" style="margin-left:8px" onclick="fetchGrader()" disabled>Grade Episode</button>
    </div>
  </div>

  <!-- Right: Output -->
  <div>
    <div class="panel" style="margin-bottom:16px">
      <h2>Vulnerable Code Snippet</h2>
      <div class="output" id="code-snippet-display">(hidden until inspected)</div>
    </div>
    <div class="panel" style="margin-bottom:16px">
      <h2>Scanner Hint</h2>
      <div class="output" id="scanner-hint-display">(hidden until inspected)</div>
    </div>
    <div class="panel">
      <h2>Score</h2>
      <div id="score-display">Run an episode and click Grade to see your score.</div>
      <div class="score-bar" id="score-bar" style="display:none">
        <div class="score-track">
          <div class="score-fill" id="score-fill" style="width:0%;background:#3b82f6">0%</div>
        </div>
      </div>
      <div id="score-breakdown" style="margin-top:12px;font-size:0.8rem;color:#94a3b8"></div>
    </div>
  </div>
</div>

<script>
let currentTaskId = null;
let stepCount = 0;

document.addEventListener('DOMContentLoaded', fetchTasks);

async function fetchTasks() {
    const res = await fetch('/tasks');
    const data = await res.json();
    const taskButtons = document.getElementById('task-buttons');
    taskButtons.innerHTML = '';
    data.tasks.forEach(task => {
        const button = document.createElement('button');
        button.className = 'btn btn-primary';
        button.textContent = `${task.id} (${task.difficulty})`;
        button.onclick = () => resetEnv(task.id);
        taskButtons.appendChild(button);
    });
}

async function resetEnv(taskId) {
  currentTaskId = taskId;
  const taskButtons = document.querySelectorAll('#task-buttons .btn');
  const execBtn = document.getElementById('execute-btn');
  const gradeBtn = document.getElementById('grade-btn');
  
  taskButtons.forEach(b => b.disabled = true);
  execBtn.disabled = true;
  gradeBtn.disabled = true;

  try {
    const res = await fetch('/reset', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({task_id: taskId})
    });

    if (!res.ok) {
      const err = await res.json();
      alert('Error resetting environment: ' + JSON.stringify(err, null, 2));
      return;
    }

    const data = await res.json();
    stepCount = 0;
    updateUI(data.observation);
    document.getElementById('score-display').textContent = 'Grade will appear after you take actions.';
    document.getElementById('score-bar').style.display = 'none';
    document.getElementById('score-breakdown').textContent = '';
    execBtn.disabled = false;
    gradeBtn.disabled = false;
  } catch (e) {
    alert('A critical error occurred: ' + e.message);
  } finally {
    taskButtons.forEach(b => b.disabled = false);
  }
}

async function takeAction() {
  const execBtn = document.getElementById('execute-btn');
  const gradeBtn = document.getElementById('grade-btn');
  execBtn.disabled = true;
  gradeBtn.disabled = true;

  const actionType = document.getElementById('action-select').value;
  const argsStr = document.getElementById('action-args').value || '{}';
  
  let action_payload = { action_type: actionType };
  if (argsStr) {
      try {
          const args = JSON.parse(argsStr);
          if (actionType === 'classify') action_payload.vulnerability_type = args.vulnerability_type;
          if (actionType === 'apply_patch') action_payload.patch_id = args.patch_id;
      } catch (e) {
          alert('Invalid JSON in arguments: ' + e.message);
          execBtn.disabled = false;
          gradeBtn.disabled = false;
          return;
      }
  }

  try {
    const res = await fetch('/step', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ action: action_payload })
    });
    
    if (!res.ok) {
      const err = await res.json();
      alert('Error taking action: ' + JSON.stringify(err, null, 2));
      return;
    }
    
    const data = await res.json();
    stepCount++;
    updateUI(data.observation);
    if (data.done) {
        fetchGrader();
    }
  } catch (e) {
    alert('A critical error occurred: ' + e.message);
  } finally {
    execBtn.disabled = false;
    gradeBtn.disabled = false;
  }
}

async function fetchGrader() {
    if (!currentTaskId) return;
    const res = await fetch(`/grader?task_id=${currentTaskId}`);
    const grade = await res.json();
    updateGraderUI(grade);
}

function updateUI(obs) {
    if (!obs) return;
    document.getElementById('step-count').textContent = stepCount;
    if (obs.status_message) {
        document.getElementById('observation-display').textContent = obs.status_message;
    }
    if (obs.code_snippet) {
        document.getElementById('code-snippet-display').textContent = obs.code_snippet;
    }
    if (obs.scanner_hint) {
        document.getElementById('scanner-hint-display').textContent = obs.scanner_hint;
    }

    const actionSelect = document.getElementById('action-select');
    const argsLabel = document.getElementById('args-label');
    const argsInput = document.getElementById('action-args');

    if (actionSelect.value === 'classify') {
        argsLabel.textContent = 'Vulnerability Type';
        argsInput.placeholder = 'e.g., "sql_injection"';
    } else if (actionSelect.value === 'apply_patch') {
        argsLabel.textContent = 'Patch ID';
        let placeholder = 'e.g., "parameterized_query"';
        if (obs.patch_options && obs.patch_options.length > 0) {
            placeholder = obs.patch_options.map(p => p.id).join(', ');
        }
        argsInput.placeholder = `Options: ${placeholder}`;
    } else {
        argsLabel.textContent = 'Arguments';
        argsInput.placeholder = '{}';
    }
}

function updateGraderUI(grade) {
  const pct = Math.round(grade.score * 100);
  const color = pct >= 95 ? '#10b981' : pct >= 50 ? '#f59e0b' : '#ef4444';
  
  document.getElementById('score-display').innerHTML = 
    `<strong style="font-size:2rem;color:${color}">${pct}%</strong> — ${grade.passed ? '✅ Passed' : '❌ Failed'}`;
  
  document.getElementById('score-bar').style.display = 'block';
  const fill = document.getElementById('score-fill');
  fill.style.width = pct + '%';
  fill.style.background = color;
  fill.textContent = pct + '%';
  
  const breakdown = grade.checks.map(c => `${c.name}: ${c.passed ? '✅' : '❌'}`).join(' | ');
  document.getElementById('score-breakdown').textContent = `Checks: ${breakdown}\n\nMessage: ${grade.message}`;
}

document.getElementById('action-select').addEventListener('change', () => {
    const actionSelect = document.getElementById('action-select');
    const argsLabel = document.getElementById('args-label');
    const argsInput = document.getElementById('action-args');
    if (actionSelect.value === 'classify') {
        argsLabel.textContent = 'Vulnerability Type';
        argsInput.placeholder = 'e.g., {"vulnerability_type": "sql_injection"}';
    } else if (actionSelect.value === 'apply_patch') {
        argsLabel.textContent = 'Patch ID';
        argsInput.placeholder = 'e.g., {"patch_id": "parameterized_query"}';
    } else {
        argsLabel.textContent = 'Arguments';
        argsInput.placeholder = '{}';
    }
});

</script>
</body>
</html>"""


def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI app with uvicorn."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def main() -> None:
    """CLI entry point exposed via the project script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
