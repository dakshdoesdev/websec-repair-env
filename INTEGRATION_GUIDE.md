# Integration Guide: Using Trained Models in Your RL Loop

This guide shows how to integrate your fine-tuned Qwen2.5-Coder model with the WebSec Repair environment for end-to-end RL training.

## Architecture

```
┌─────────────────────────────────────────┐
│   WebSecRepairEnv (Environment)         │
│  - State: code, scanner hint             │
│  - Action: inspect/classify/patch/verify │
│  - Reward: composite score                │
└──────────────────┬──────────────────────┘
                   │
                   ↓
        ┌──────────────────────┐
        │ WebSecRepairAgent    │
        │ (Fine-tuned Model)   │
        │                      │
        │ Takes observation    │
        │ → Returns action     │
        └──────────────────────┘
```

## Step 1: Load Model After Training

```python
from training.inference import WebSecRepairAgent
from websec_repair_env import WebSecRepairEnv, WebSecRepairAction

# Load fine-tuned model
agent = WebSecRepairAgent("./qwen-websec-finetuned")

# Initialize environment
env = WebSecRepairEnv(base_url="http://localhost:8000")
```

## Step 2: Convert Environment Observations to Agent Input

```python
def observation_to_prompt(obs):
    """Convert environment observation to agent-friendly prompt."""
    return {
        "task_id": obs.get("task_id", ""),
        "instruction": obs.get("instruction", ""),
        "code_snippet": obs.get("code_snippet", ""),
        "scanner_hint": obs.get("scanner_hint", ""),
    }
```

## Step 3: Run Episode with Trained Agent

```python
def run_episode(env, agent, task_id="sqli_login"):
    """Run one episode with trained agent."""
    obs = env.reset(task_id=task_id)
    done = False
    episode_reward = 0
    steps = 0
    
    while not done and steps < 10:
        # Get agent's recommended action
        prompt_data = observation_to_prompt(obs)
        action_dict = agent.get_action(**prompt_data)
        
        # Convert to WebSecRepairAction
        action = WebSecRepairAction(
            action_type=action_dict.get("action_type", "inspect"),
            vulnerability_type=action_dict.get("vulnerability_type"),
            patch_id=action_dict.get("patch_id"),
        )
        
        # Execute in environment
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        steps += 1
        
        print(f"Step {steps}: {action.action_type} → Reward: {reward:.4f}")
    
    return {
        "task_id": task_id,
        "total_reward": episode_reward,
        "steps": steps,
        "success": done,
        "grader_score": info.get("grader", {}).get("score", 0),
    }

# Run episode
result = run_episode(env, agent, task_id="sqli_login")
print(f"\nEpisode Result: {result}")
```

## Step 4: Training Loop (Multiple Episodes)

```python
from visualization import MetricsTracker

# Initialize metrics tracker
tracker = MetricsTracker(run_name="trained_agent_v1")

# Run multiple episodes
for episode in range(10):
    result = run_episode(env, agent, task_id="sqli_login")
    
    # Track metrics
    tracker.record_episode(
        episode=episode + 1,
        task_id=result["task_id"],
        cumulative_reward=result["total_reward"],
        success=result["success"],
        steps=result["steps"],
    )
    
    print(f"Episode {episode + 1}/10: Reward={result['total_reward']:.4f}, Success={result['success']}")

# Save metrics
import json
metrics_file = Path("trained_run/metrics.json")
metrics_file.parent.mkdir(parents=True, exist_ok=True)
with open(metrics_file, "w") as f:
    json.dump(tracker.to_dict(), f, indent=2)
print(f"✓ Metrics saved to {metrics_file}")
```

## Step 5: Evaluate and Compare

```python
from visualization import ComparisonPlotter

# Load baseline and trained metrics
baseline_metrics = json.load(open("baseline_run/metrics.json"))
trained_metrics = json.load(open("trained_run/metrics.json"))

# Generate comparison
plotter = ComparisonPlotter()
plotter.add_run("Baseline (Rule-based)", baseline_metrics)
plotter.add_run("Trained (Qwen2.5-Coder)", trained_metrics)

# Create plots
plotter.plot_success_rate_comparison("success_rate.png")
plotter.plot_reward_trajectory_comparison("reward_trajectory.png")
plotter.plot_average_metrics_comparison("average_metrics.png")
plotter.plot_accuracy_comparison("accuracy_comparison.png")

print("✓ Comparison plots generated")
```

## Complete Example Script

```python
#!/usr/bin/env python3
"""Full training loop with fine-tuned model."""

import json
from pathlib import Path

from training.inference import WebSecRepairAgent
from visualization import MetricsTracker, ComparisonPlotter
from websec_repair_env import WebSecRepairEnv, WebSecRepairAction


def observation_to_prompt(obs):
    return {
        "task_id": obs.get("task_id", ""),
        "instruction": obs.get("instruction", ""),
        "code_snippet": obs.get("code_snippet", ""),
        "scanner_hint": obs.get("scanner_hint", ""),
    }


def run_episode(env, agent, task_id):
    obs = env.reset(task_id=task_id)
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < 10:
        action_dict = agent.get_action(**observation_to_prompt(obs))
        action = WebSecRepairAction(
            action_type=action_dict.get("action_type", "inspect"),
            vulnerability_type=action_dict.get("vulnerability_type"),
            patch_id=action_dict.get("patch_id"),
        )
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
    
    return {
        "task_id": task_id,
        "total_reward": total_reward,
        "steps": steps,
        "success": done,
    }


def main():
    # Load model
    print("Loading trained model...")
    agent = WebSecRepairAgent("./qwen-websec-finetuned")
    
    # Initialize environment
    print("Initializing environment...")
    env = WebSecRepairEnv(base_url="http://localhost:8000")
    
    # Run episodes
    print("Running 10 episodes...")
    tracker = MetricsTracker(run_name="trained_agent_final")
    
    for episode in range(10):
        result = run_episode(env, agent, task_id="sqli_login")
        tracker.record_episode(
            episode=episode + 1,
            task_id=result["task_id"],
            cumulative_reward=result["total_reward"],
            success=result["success"],
            steps=result["steps"],
        )
        print(f"  Episode {episode + 1}: Reward={result['total_reward']:.4f}, Success={result['success']}")
    
    # Save metrics
    metrics_file = Path("trained_run_final/metrics.json")
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w") as f:
        json.dump(tracker.to_dict(), f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"✓ Metrics saved to {metrics_file}")
    
    # Cleanup
    agent.unload()
    env.close()


if __name__ == "__main__":
    main()
```

Save as `run_trained_agent.py` and run:

```bash
# Terminal 1: Start environment server
python -m websec_repair_env.server.app

# Terminal 2: Run training loop
python run_trained_agent.py
```

## Key Differences: Baseline vs Trained Agent

### Baseline (Rule-based)
```python
def choose_vulnerability(task_id, scanner_hint):
    # Simple keyword matching
    if "sql" in scanner_hint.lower():
        return "sql_injection"
    return "unknown"
```

- Fixed logic
- No learning
- Consistent but limited

### Trained (LLM)
```python
agent = WebSecRepairAgent("./qwen-websec-finetuned")
action = agent.get_action(task_id, instruction, code, hint)
# Model-generated: learns patterns, generalizes
```

- Learns from training data
- Adapts to new patterns
- Better accuracy on diverse code

## Monitoring Training

```python
# During training, log metrics
for episode in range(100):
    result = run_episode(env, agent, task_id)
    
    # Log to tracker
    tracker.record_episode(
        episode=episode,
        task_id=result["task_id"],
        cumulative_reward=result["total_reward"],
        success=result["success"],
        steps=result["steps"],
        vulnerability_correct=result.get("vuln_correct", False),
        patch_correct=result.get("patch_correct", False),
    )
    
    # Every 10 episodes, save checkpoint
    if episode % 10 == 0:
        checkpoint_file = f"checkpoints/metrics_ep{episode}.json"
        with open(checkpoint_file, "w") as f:
            json.dump(tracker.to_dict(), f)
```

## Expected Results

With Qwen2.5-Coder-1.5B fine-tuned:

| Metric | Baseline | Trained | Improvement |
|--------|----------|---------|-------------|
| Task Success Rate | 80-90% | 85-95% | +5-15% |
| Vulnerability Classification | 20-40% | 60-80% | +40% |
| Patch Selection Accuracy | 70-80% | 75-90% | +5-20% |
| Avg Steps per Task | 5-6 | 4-5 | -20% |
| Total Episode Reward | 2.5-3.5 | 3.5-4.5 | +40% |

Results depend on:
- Training data quality
- Model size (1.5B vs 3B)
- Fine-tuning epochs
- Task complexity

## Troubleshooting

### "Model memory error during inference"
```python
# Unload model periodically
if episode % 20 == 0:
    agent.unload()
    agent = WebSecRepairAgent("./qwen-websec-finetuned")
```

### "Low accuracy despite training"
1. Check training data quality
2. Verify model loaded correctly
3. Increase fine-tuning epochs
4. Try larger model (3B)

### "Slow inference"
1. Use quantized model
2. Reduce max_new_tokens (currently 256)
3. Batch multiple inferences
4. Run on faster GPU

## Next Steps

1. **Collect more training data**: Log real trajectories
2. **Fine-tune longer**: Try 5-10 epochs
3. **Larger model**: Train 3B variant
4. **Production deployment**: Quantize and containerize
5. **Multi-task learning**: Train on multiple vulnerability types

---

See [TRAINING_GUIDE.md](./TRAINING_GUIDE.md) for model training instructions.
