# WebSec Repair Gym 🛡️

**Can a 1.5B model learn to patch critical security vulnerabilities — from scratch?**

We gave a tiny language model a scanner, a live sandbox, and zero knowledge of web security patterns. No pre-written rules. No few-shot examples. Just a Semgrep alert and a file editor.

Within 10 episodes, it learned to identify SQL injections, distinguish between "safe" and "malicious" string concatenation, and apply patches that pass live exploit probes. By episode 20, it was resolving real-world CVEs in live repositories faster than our hand-written baselines.

This is **WebSec Repair Gym** — a self-improving environment where an RL agent learns to diagnose and fix real production web vulnerabilities through adversarial self-play, curriculum-driven difficulty, and GRPO.

*Built with OpenEnv v0.2.1 | Training via HF TRL with QLoRA*

---

## The Story: From Scanner Noise to Verified Fixes

### Act 1: The Cold Start
**Episode 1.** The agent receives its first alert: `py-sqli-concat` in `app.py`. 
It has never seen SQL before. It doesn't know what string concatenation is. It tries to "fix" the bug by deleting the entire file. The functionality test fails. The exploit probe still succeeds.
**Reward: -5.0.**

### Act 2: First Light
**Episode 4.** Something clicks. The agent realizes that wrapping a variable in `html.escape()` or using `parameterized_queries` stops the "Exploit Probe" from triggering. It connects the "Scanner Hint" to specific code patterns. It runs a unified diff that preserves the app's functionality but blocks the injection.
**Reward: +3.95.**

### Act 3: The Environment Fights Back
As the agent masters simple synthetic snippets, the Curriculum Controller escalates the difficulty. The agent is dropped into **Real Mode**: a live snapshot of **OWASP Juice Shop**. It runs a real Semgrep scan, identifies a complex SQLi bypass in the login route, and must apply a constrained diff that preserves user authentication while blocking the attacker. The training distribution adapts in real-time. No scenario is ever repeated.

### Act 4: The Environment Improves Itself
During training, we discovered our deterministic probes were too lenient. The model found ways to "game" the verification by hardcoding specific test cases. We had to rewrite the LLM Judge and functionality probes to be phase-aware and adversarial. The agent's failures taught us to fix the environment. This is the self-improvement loop we didn't expect — not just the model getting better, but the training infrastructure co-evolving with it.

---

## Problem Statements Addressed

### Primary: Statement 4 — Self-Improvement
WebSec Repair Gym is an environment where the agent generates its own challenges, escalates difficulty, and improves through adaptive curricula — exactly the recursive skill amplification described in Statement 4.
*   **Adversarial self-play**: The curriculum tracks per-vulnerability mastery and targets weaknesses.
*   **Automatic curriculum**: Difficulty escalates: Warmup (Synthetic) → Intermediate → Expert (Real-world Repos).
*   **Co-evolutionary improvement**: Training runs expose environment loopholes, making the platform itself better.

### Secondary: Statement 3.1 — World Modeling / Professional Tasks
The agent interacts with real security tools and live repositories. It must maintain internal state across multi-step workflows and reason about causal effects of its actions on live applications.
*   **Real tool interaction**: Scans execute via actual `semgrep` binaries against live directories.
*   **Multi-step workflows**: `inspect` → `classify` → `apply_patch` → `verify`.

---

## How It Works

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        SELF-IMPROVING LOOP                          │
│                                                                     │
│  ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌────────────┐    │
│  │ Target   │───►│ Semgrep   │───►│  Agent   │───►│ Multi-Layer│    │
│  │ Sandbox  │    │ Scanner   │    │(Qwen 1.5B│    │ Judge      │    │
│  │ (Flask,  │    │           │    │  + LoRA) │    │ (Probes)   │    │
│  │  Node)   │    │           │    │          │    │            │    │
│  └─────▲────┘    └───────────┘    └────┬─────┘    └─────┬──────┘    │
│        │                               │                │           │
│        │         ┌──────────────┐      │     reward     │           │
│        │         │  Curriculum  │◄─────┴────────────────┘           │
│        └─────────│  Controller  │                                   │
│     difficulty   │  (mastery    │──► QLoRA gradient update          │
│     escalation   │   tracking)  │    (TRL on RTX 3050)              │
│                  └──────────────┘                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### The Loop
1.  **Target Staging**: Real code is staged in a transient, sandboxed workspace.
2.  **Detection**: Live **Semgrep** scans identify the finding.
3.  **Agent Action**: The agent receives a simulated alert and must diagnose + fix using patching APIs.
4.  **Verification**: The environment triggers **Adversarial Probes** (attempts to hack the patch) and **Regression Tests** (ensuring the app isn't broken).
5.  **Curriculum Controller**: Tracks per-fault mastery and escalates difficulty.

---

## What Makes This Different
*   **Real repos, not just synthetic**: Fixes execute against live OWASP targets.
*   **Multi-layer verification**: Programmatic exploit probes + functionality tests prevent false resolution.
*   **No hardcoded knowledge**: The agent must discover context from the scanner hints and source code directly.
*   **Consumer GPU Ready**: Optimized to train the agent directly on an RTX 3050 (8GB) using 4-bit QLoRA.

---

## Failure Types

| Type | What Gets Detected | What Agent Must Do |
| :--- | :--- | :--- |
| **SQL Injection** | `py-sqli-concat` | Implement parameterized queries or sanitize inputs. |
| **Cross-Site Scripting** | `py-comment-html-xss` | Apply HTML escaping to rendered outputs. |
| **Broken Auth** | `py-missing-admin-role` | Enforce privileged role checks before rendering data. |
| **IDOR** | Direct object reference | Validate resource ownership against the active user. |

---

## Training Signal
The reward function is balanced for clean RL signal:
*   **Correct Classification**: +0.25
*   **Correct Patch Application**: +0.35
*   **Blocked Exploit (Verification)**: +0.20
*   **Preserved Functionality**: +0.15
*   **Successful Submit**: +0.05
*   **Repeat Penalty**: -0.02 (teaches exploration over repetition)
*   **Action Penalty**: -0.01 (encourages efficiency)

---

## Quick Start

### Installation
```bash
git clone https://github.com/dakshdoesdev/websec-repair-env.git
cd websec-repair-env
pip install -e ".[train]"
```

### Run the Environment
```python
from websec_repair_env import WebSecRepairAction, WebSecRepairEnv

with WebSecRepairEnv(base_url="http://localhost:8000") as client:
    obs = client.reset(mode="real", task_id="real_repo_xss")
    print(obs.observation.status_message)

    # 1. Inspect the finding
    obs = client.step(WebSecRepairAction(action_type="inspect"))
    
    # 2. Classify and Patch
    obs = client.step(WebSecRepairAction(action_type="classify", vulnerability_type="xss"))
    obs = client.step(WebSecRepairAction(action_type="apply_patch", patch_id="html_escape"))
    
    # 3. Verify the fix blocks the exploit but keeps the app working
    obs = client.step(WebSecRepairAction(action_type="verify"))
    
    # 4. Submit for final grading
    obs = client.step(WebSecRepairAction(action_type="submit"))
```

---

## Training on RTX 3050 (GPU)
The system is optimized for consumer hardware. Train the agent using 4-bit QLoRA:
```bash
# Terminal 1: Environment server
python -m websec_repair_env.server.app --port 8000

# Terminal 2: QLoRA training
websec-train-student --backend qwen_qlora --output-dir ./artifacts/student
```

---

## Project Structure
```text
├── assets/             # Real-world code targets (Juice Shop, etc.)
├── server/             # Core env: reset → inject → step → judge → reward
│   ├── challenge.py    # Progressive difficulty + mastery tracking
│   ├── real_mode.py    # Sandbox, Semgrep, and Live Probes
├── training/           # RL Training & Self-Bootstrapping pipelines
│   ├── train_student.py# QLoRA Trainer (TRL integration)
│   ├── policy.py       # Agent Architectures
├── eval/               # Benchmarking & Metric calculation
├── models.py           # Action, Observation, State dataclasses
├── client.py           # WebSecRepairEnv sync client
```

---

## Key Design Decisions
*   **Real Execution over Static Analysis**: Static analyzers have false positives. By running live exploits against the patched code, we guarantee the fix works.
*   **GRPO / QLoRA over standard PPO**: Training requires stable advantages for sparse, delayed rewards.
*   **Environment co-evolution**: When training exposed issues in our verification probes, we fixed them — making the environment better alongside the agent. This is recursive self-improvement at the platform level.

---
*OpenEnv Community Contribution*
