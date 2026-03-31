# Learning Plan: Incident Response Commander (IRC) Gym

This document provides a structured plan to understand and engage with the "Incident Response Commander" (IRC) project, which was a submission for the OpenEnv Hackathon.

## 1. Project Overview

The core of this project is the **Incident Response Commander (IRC)**, a reinforcement learning environment that simulates real-world production outages. The goal is to train an AI agent to act as a Site Reliability Engineer (SRE), capable of diagnosing and resolving complex infrastructure failures under time pressure.

- **Objective**: Train an AI to handle production incidents.
- **Key Challenge**: The agent must perform a multi-step workflow: `investigate` -> `diagnose` -> `remediate` -> `communicate`.
- **Unique Feature**: The environment includes **adversarial prompt injection** in its log files, testing the agent's ability to distinguish between trusted metrics and untrusted, potentially malicious, log data.

---

## 2. Core Concepts (from Research Docs & README)

Before diving into the code, it's essential to understand the SRE principles the environment is built on.

| Concept | Description | Relevance to IRC |
| :--- | :--- | :--- |
| **Golden Signals** | The four key metrics for monitoring a user-facing system: Latency, Traffic, Errors, and Saturation. | The environment's `metrics` observation directly models these. |
| **Error Budgets / SLOs** | A budget for how much a service is allowed to fail over a period. Exceeding it triggers alerts. | The `slo_context` in the observation space is a direct implementation of this. |
| **Cascading Failures** | When a failure in one system triggers failures in other dependent systems. | **Task 3 (Midnight Stampede)** is a classic example of this pattern. |
| **Red Herrings** | Distractor alerts or metrics that seem important but are not related to the root cause. | **Task 2 (Phantom Latency)** is explicitly designed around this concept. |
| **Load Shedding** | Intentionally dropping a percentage of traffic to prevent a total system collapse during overload. | This is the optimal first action in **Task 3**. |
| **Prompt Injection** | An attack where malicious instructions are embedded in data that an LLM will process. | A core security challenge in the IRC environment. The agent must learn to ignore instructions found in logs. |

---

## 3. Codebase Analysis: How It Works

The project is a standard FastAPI web server that exposes an OpenEnv-compliant API.

### `src/server/app.py`
- **Purpose**: This is the main web server. It uses FastAPI to create the API endpoints (`/reset`, `/step`, `/grade`).
- **Key Logic**: It initializes the `IncidentResponseCommander` environment and exposes its methods over HTTP. It also serves the interactive web UI.

### `src/env/commander_env.py` (The "Gym")
- **Purpose**: This file contains the complete logic for the reinforcement learning environment.
- **Key Components**:
    - `IncidentResponseCommander` class: Manages the state of the simulation.
    - `reset()`: Initializes one of the three tasks.
    - `step()`: The most important function. It takes an agent's action, updates the simulation state, and calculates the reward.
    - `grade()`: Calculates the final score based on whether the incident was resolved and how optimally it was handled.

### `inference.py` (The "Agent")
- **Purpose**: This is a baseline agent that uses an LLM (like GPT-4o-mini) to interact with the environment.
- **Key Logic**:
    1.  It constructs a **system prompt** that gives the LLM its "SRE persona" and critical rules.
    2.  It calls the `/reset` endpoint to start an episode.
    3.  It enters a loop:
        - It builds a **user prompt** from the current observation (alerts, metrics, logs).
        - It sends this to the LLM API.
        - It **parses the JSON action** from the LLM's response.
        - It sends the action to the `/step` endpoint.
    4.  The loop continues until the episode is `done`.

---

## 4. Step-by-Step Learning Path

Follow these steps to get a complete understanding of the project.

### Step 1: Run the Environment (15 minutes)
First, get a feel for the environment by interacting with it manually.
1.  **Start the server**: `uvicorn src.server.app:app --host 0.0.0.0 --port 7860`
2.  **Open the Web UI**: Go to `http://localhost:7860` in your browser.
3.  **Play Through Task 1**:
    - Click `Task 1 — Easy`.
    - Manually take actions. Try to `query_logs` first. You should see messages about `CrashLoopBackOff`.
    - Try to `rollback_deployment`. The incident should resolve.
    - Click `Grade Episode` to see your score.

### Step 2: Run the Baseline Agent (15 minutes)
Next, see how an AI agent performs.
1.  **Set Environment Variables**:
    ```bash
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4o-mini"
    export HF_TOKEN="your-openai-api-key"
    export ENV_URL="http://localhost:7860"
    ```
2.  **Run Inference**:
    ```bash
    python inference.py
    ```
3.  **Observe the Output**: Watch the terminal as the LLM makes decisions. Note its reasoning. See if it can solve all three tasks.

### Step 3: Code Deep-Dive (1 hour)
Now, read the code in this specific order to understand the logic flow.

1.  **`src/env/models.py`**: Understand the data structures. What is in an `IRCAction`? What is in an `IRCObservation`?
2.  **`src/server/app.py`**: See how the FastAPI routes map directly to the environment's methods.
3.  **`src/env/commander_env.py`**:
    - Read the `reset()` function first. See how `TASKS` are loaded.
    - Read the `step()` function. Pay close attention to the `if/else` blocks for each `tool_name`. This is where the simulation logic lives.
    - Read the `grade()` function to understand how the final score is calculated.
4.  **`inference.py`**:
    - Read the `SYSTEM_PROMPT`. This is the "soul" of the agent.
    - Read the `run_episode` function to see the agent's loop in action.

### Step 4: Extend and Improve (Ongoing)
Once you understand the system, you can try to improve it:
- **Improve the Prompt**: Can you write a better system prompt in `inference.py` to get a higher score?
- **Add a New Tool**: Add a new action (e.g., `get_deployment_yaml`) to `commander_env.py` and `inference.py`.
- **Create a New Task**: Define `TASK_4` in `commander_env.py` with a new, more complex failure scenario.
