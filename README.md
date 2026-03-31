# WebSec Repair Gym

### Training AI agents to detect, exploit, and repair real-world web vulnerabilities

---

## Overview

**WebSec Repair Gym** is an OpenEnv-based reinforcement learning environment where an AI agent learns to **identify, patch, and verify security vulnerabilities in real codebases**.

Unlike static analysis tools or rule-based systems, this framework forces the agent to operate in a **closed-loop workflow**:

> **inspect → classify → patch → verify → submit**

Every action is validated through **real exploit probes and functionality tests**, ensuring that fixes are not only correct — but safe.

---

## Core Idea

Most AI security tools today:

* detect vulnerabilities
* suggest fixes
* but **never verify if the fix actually works**

This project solves that.

We train an agent that must:

* understand the vulnerability
* apply a correct patch
* prove the exploit no longer works
* ensure the application still functions

If it fails any step → it is penalized.

---

## What This Actually Is

This is **not just a model**.

It is a **self-improving training system** built around:

* Real detection tools (Semgrep)
* Adversarial verification (exploit probes)
* Deterministic grading
* Reward-driven learning
* Curriculum-based difficulty escalation

The agent improves **across episodes**, not by guessing — but through **measurable feedback loops**.

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

## Training Loop

```text
Target Codebase
      |
      v
Semgrep Scanner
      |
      v
Agent (LLM + Policy)
      |
      v
Patch Application
      |
      v
Verification Layer
 (Exploit + Functionality)
      |
      v
Reward + Feedback
      |
      v
Curriculum Controller
      |
      v
Next Episode (Harder)
```

---

## Modes

### 1. Deterministic Mode (Stable / Evaluation)

* Predefined vulnerable scenarios
* Fully controlled grading
* Fast and reliable
* Used for benchmarking and validation

Tasks:

* SQL Injection (login)
* XSS (comment rendering)
* Broken Auth (admin route)

---

### 2. Real Mode (Advanced)

* Runs on **real codebases**
* Uses **Semgrep** for detection
* Applies patches to actual files
* Re-validates using:

  * exploit probes
  * functional checks

Examples:

* OWASP Juice Shop (SQLi)
* Real Flask apps (XSS / auth bugs)

---

## What the Agent Learns

| Skill      | Description                               |
| ---------- | ----------------------------------------- |
| Detection  | Interpret scanner output and code context |
| Reasoning  | Identify root cause of vulnerability      |
| Repair     | Apply safe, minimal patches               |
| Validation | Confirm exploit is blocked                |
| Safety     | Ensure functionality is preserved         |

---

## Reward System

The agent is graded objectively:

| Component                   | Reward             |
| --------------------------- | ------------------ |
| Correct Classification      | +0.25              |
| Correct Patch               | +0.35              |
| Exploit Blocked             | +0.20              |
| Functionality Preserved     | +0.15              |
| Successful Submission       | +0.05              |
| Inefficient / Wrong Actions | Negative penalties |

This prevents:

* guessing
* brute force
* unsafe fixes

---

## What Makes This Different

### 1. Real Verification

Not just "fix suggested" — the system **attacks the patch**.

### 2. Closed-Loop Learning

The agent must complete the full cycle:

> detect → fix → prove → validate

### 3. Environment Co-Evolution

Failures expose weaknesses in:

* reward design
* verification logic
* test coverage

The environment improves alongside the agent.

### 4. Consumer-Grade Training

No cluster required. Built to run on **local GPUs**.

---

## Limitations (Honest)

* Not fully autonomous real-world scanning yet
* Patch space is template-based (not free-form editing)
* True online continual learning (per-run weight updates) is **not implemented yet**

---

## Future Work

* Memory system (experience replay across runs)
* Multi-agent coordination (scanner + patcher roles)
* Real-time CI/CD integration
* Expanded vulnerability coverage (IDOR, SSRF, RCE)
* Fully autonomous patch generation

---

## Quick Start

### Installation

```bash
git clone https://github.com/your-username/websec-repair-env.git
cd websec-repair-env
pip install -e ".[train]"
```

### Run the Environment

```python
from websec_repair_env import WebSecRepairAction, WebSecRepairEnv

with WebSecRepairEnv(base_url="http://localhost:8000") as env:
    obs = env.reset(task_id="sqli_login")

    env.step(WebSecRepairAction(action_type="inspect"))
    env.step(WebSecRepairAction(action_type="classify", vulnerability_type="sql_injection"))
    env.step(WebSecRepairAction(action_type="apply_patch", patch_id="parameterized_query"))
    env.step(WebSecRepairAction(action_type="verify"))
    env.step(WebSecRepairAction(action_type="submit"))
```

---

## Training Models with QLoRA

### Quick Start: Colab (Free GPU)

Train **Qwen2.5-Coder-1.5B** on Google Colab with free GPU (15-30 minutes):

```bash
# In Colab notebook
!git clone https://github.com/your-org/websec-repair-env.git
%cd websec-repair-env
!pip install -q -r training_requirements.txt
!python training/qlora_train.py --output-dir ./qwen-websec-finetuned --epochs 3
```

→ See [TRAINING_GUIDE.md](./TRAINING_GUIDE.md) for detailed instructions

### What We Train On

**Data mix (recommended):**
- 60% your environment trajectories
- 25% SWE-bench Lite (real bug fixes)
- 15% PrimeVul/Devign (vulnerability recognition)

**Recommended models:**
- **Qwen2.5-Coder-1.5B-Instruct** ← start here
- **Qwen2.5-Coder-3B-Instruct** (better accuracy, slower)

**Training method:**
- 4-bit quantization (QLoRA)
- LoRA rank 16, 32 target layers
- Learning rate 2e-4, 3 epochs
- 4 batch size, 4 gradient accumulation

### Use Trained Model

After training:

```python
from training.inference import WebSecRepairAgent

agent = WebSecRepairAgent("./qwen-websec-finetuned")
action = agent.get_action(
    task_id="sqli_login",
    instruction="Fix SQL injection",
    code_snippet="...",
    scanner_hint="SQL injection detected"
)
```

### Local Training

Train on consumer GPU (RTX 3050+):

```bash
# Terminal 1: Environment server
python -m websec_repair_env.server.app --port 8000

# Terminal 2: Fine-tune model
python training/qlora_train.py \
  --output-dir ./qwen-websec-local \
  --model-id Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --epochs 3 \
  --batch-size 4
```

---

## Architecture

```text
├── assets/             # Real-world targets (Juice Shop, Flask apps)
├── server/
│   ├── app.py          # OpenEnv API
│   ├── challenge.py    # Task + grading logic
│   ├── real_mode.py    # Semgrep + sandbox + probes
│   └── environment.py  # Core step/reset logic
├── models.py           # State, Action, Observation
├── client.py           # Sync client
├── inference.py        # Baseline agent
├── training/           # RL + QLoRA pipelines
```

---

## Final Positioning

This project is best described as:

> **A reinforcement learning environment for training AI agents to safely repair real-world software vulnerabilities through verified feedback loops.**

Not a demo.
Not a script.
A **learning system**.

---

*OpenEnv Community Contribution*
