# WebSec Repair Environment - Visualization & Metrics Guide

## Overview

The visualization system provides **side-by-side comparison** of base model vs. trained model performance with matplotlib graphs and an interactive HTML report.

## Quick Start

### 1. Run the Demo

See sample visualization with pre-generated metrics:

```bash
python demo_visualization.py
```

This generates:
- `visualization_results/report.html` - Interactive report
- `visualization_results/*.png` - 4 comparison plots
- `visualization_results/*.json` - Raw metrics data

Open `visualization_results/report.html` in your browser to view the full report.

### 2. Track Your Own Runs

#### Using the Enhanced Inference Script

```bash
# Run 10 episodes and track metrics
python inference_with_metrics.py --episodes 10 --metrics --output-dir my_results

# With custom server
python inference_with_metrics.py --base-url http://localhost:8000 --task sqli_login --episodes 5 --metrics
```

#### Programmatic Usage

```python
from visualization import MetricsTracker, ComparisonPlotter

# Create tracker
tracker = MetricsTracker(run_name="my_run")

# After each episode, record metrics
tracker.add_episode(
    episode=1,
    task_id="sqli_login",
    cumulative_reward=0.8,
    success=True,
    steps=4,
    vulnerability_correct=True,
    patch_correct=True,
    time_seconds=2.5,
)

# Save metrics
tracker.save_json("my_metrics.json")

# Generate visualizations
plotter = ComparisonPlotter(output_dir="results")
plotter.plot_success_rate_comparison(base_tracker, trained_tracker)
```

## Key Components

### MetricsTracker

Tracks metrics for individual episodes:

```python
tracker = MetricsTracker(run_name="baseline")

# Add episode data
tracker.add_episode(
    episode=1,
    task_id="sqli_login",
    cumulative_reward=0.5,
    success=True,
    steps=4,
    vulnerability_correct=True,      # Was vulnerability correctly identified?
    patch_correct=True,               # Was correct patch chosen?
    time_seconds=2.3,
)

# Get aggregate metrics
agg = tracker.get_aggregate()
print(f"Success Rate: {agg.success_pct}%")
print(f"Avg Reward: {agg.avg_reward}")

# Save to JSON
tracker.save_json("metrics.json")
```

### ComparisonPlotter

Generates matplotlib visualizations:

```python
plotter = ComparisonPlotter(output_dir="visualization_results")

# Generate individual plots
plotter.plot_success_rate_comparison(base_tracker, trained_tracker)
plotter.plot_reward_trajectory(base_tracker, trained_tracker)
plotter.plot_average_metrics(base_tracker, trained_tracker)
plotter.plot_accuracy_comparison(base_tracker, trained_tracker)

# Generate full HTML report
report_path = plotter.generate_html_report(
    base_tracker,
    trained_tracker,
    report_title="My Training Run"
)
```

## Comparison Metrics

### What Gets Tracked

| Metric | Description |
|--------|-------------|
| **Success Rate** | % of tasks successfully solved |
| **Cumulative Reward** | Total reward per episode |
| **Steps** | Number of actions taken |
| **Vulnerability Accuracy** | % of correct vulnerability classifications |
| **Patch Accuracy** | % of correct patch selections |
| **Time** | Seconds to complete episode |

### Generated Visualizations

1. **Success Rate Comparison** - Bar chart showing % success
2. **Reward Trajectory** - Line graph over episodes
3. **Average Metrics** - Reward and steps comparison
4. **Accuracy Metrics** - Vulnerability and patch accuracy

## Example Workflow

### Step 1: Baseline Run

```python
from visualization import MetricsTracker

# Run baseline agent and collect metrics
base_tracker = MetricsTracker("baseline")

# Simulate running baseline on 10 episodes
for episode in range(1, 11):
    # ... run baseline episode ...
    base_tracker.add_episode(
        episode=episode,
        task_id="sqli_login",
        cumulative_reward=0.1,  # Low baseline reward
        success=(episode > 7),   # Low success rate
        steps=5,
        vulnerability_correct=(episode % 2 == 0),
        patch_correct=(episode > 7),
        time_seconds=2.5,
    )

base_tracker.save_json("baseline_metrics.json")
```

### Step 2: Trained Run

```python
from visualization import MetricsTracker

# Run trained agent and collect metrics
trained_tracker = MetricsTracker("trained")

for episode in range(1, 11):
    # ... run trained episode ...
    trained_tracker.add_episode(
        episode=episode,
        task_id="sqli_login",
        cumulative_reward=0.6,  # Higher reward
        success=(episode > 3),   # Higher success rate
        steps=4,
        vulnerability_correct=(episode > 2),
        patch_correct=(episode > 2),
        time_seconds=2.0,
    )

trained_tracker.save_json("trained_metrics.json")
```

### Step 3: Generate Report

```python
from visualization import MetricsTracker, ComparisonPlotter

# Load metrics
base_metrics = MetricsTracker.load_json("baseline_metrics.json")
trained_metrics = MetricsTracker.load_json("trained_metrics.json")

# Create trackers from loaded data
base_tracker = MetricsTracker("baseline")
trained_tracker = MetricsTracker("trained")

# ... repopulate episodes ...

# Generate visualization
plotter = ComparisonPlotter(output_dir="comparison_results")
report_path = plotter.generate_html_report(base_tracker, trained_tracker)

print(f"Report saved to: {report_path}")
```

## Output Files

```
visualization_results/
├── report.html              # Main interactive report
├── 01_success_rate.png      # Success rate bar chart
├── 02_reward_trajectory.png # Reward progression line graph
├── 03_average_metrics.png   # Average reward and steps
├── 04_accuracy_comparison.png # Classification accuracy
├── base_metrics.json        # Raw baseline metrics
└── trained_metrics.json     # Raw trained metrics
```

### report.html Features

- **Key Findings Section** - Summary of improvements
- **Metrics Cards** - Quick stats for both models
- **Embedded Plots** - All 4 comparison graphs
- **Responsive Design** - Works on mobile and desktop
- **Dark Theme** - Eye-friendly dark blue color scheme

## Integration with Training

To integrate with your RL training pipeline:

```python
from visualization import MetricsTracker

# During training loop
metrics = MetricsTracker("training_run_1")

for episode in range(num_episodes):
    # ... run training episode ...
    
    metrics.add_episode(
        episode=episode,
        task_id=current_task,
        cumulative_reward=episode_reward,
        success=task_completed,
        steps=action_count,
        vulnerability_correct=vuln_correct,
        patch_correct=patch_correct,
        time_seconds=elapsed_time,
    )

# After training
metrics.save_json(f"runs/run_{timestamp}.json")
```

## Customization

### Custom Metrics

Extend `EpisodeMetrics` dataclass in `visualization.py`:

```python
@dataclass
class EpisodeMetrics:
    episode: int
    task_id: str
    cumulative_reward: float
    success: bool
    steps: int
    vulnerability_correct: bool
    patch_correct: bool
    time_seconds: float
    custom_metric: float = 0.0  # Add custom field
```

### Custom Plots

Add new plotting methods to `ComparisonPlotter`:

```python
def plot_custom_metric(self, tracker1, tracker2):
    # Use matplotlib to create your plot
    fig, ax = plt.subplots()
    # ... plotting code ...
    filepath = self.output_dir / "custom_plot.png"
    plt.savefig(filepath)
    plt.close()
    return str(filepath)
```

## Performance Notes

- **Matplotlib Backend**: Uses 'Agg' for headless operation (no display needed)
- **Memory**: Stores all episode metrics in memory - efficient for up to 10,000 episodes
- **Plot Generation**: ~1-2 seconds per plot
- **HTML Report**: Generated in <1 second

## FAQ

**Q: Can I compare multiple runs?**  
A: Yes! Load multiple JSON files and generate separate comparisons for each pair.

**Q: Can I customize plot colors?**  
A: Yes, modify the color codes in `ComparisonPlotter` methods.

**Q: What's the minimum episodes needed?**  
A: At least 1 episode per tracker. Recommended: 10+ for meaningful comparisons.

**Q: Can I save plots in different formats?**  
A: Yes, modify `plt.savefig()` calls to use `.pdf`, `.jpg`, etc.

**Q: Does the system require a GPU?**  
A: No, metrics tracking and visualization use CPU only.

## Troubleshooting

**Error: "No module named 'matplotlib'"**
```bash
pip install matplotlib
```

**Error: "Failed to save plot"**
- Ensure `output_dir` directory exists
- Check write permissions
- The `ComparisonPlotter` creates the directory automatically

**Plots look blurry**
- Increase `dpi` parameter in `savefig()` calls (default: 150)

**Report doesn't display images**
- Ensure PNG files are in the same directory as HTML
- Use absolute paths if deploying to web server

## Next Steps

1. **Run demo**: `python demo_visualization.py`
2. **Integrate tracking**: Use `MetricsTracker` in your training loop
3. **Generate reports**: Use `ComparisonPlotter` after training
4. **Analyze results**: Open `report.html` to review improvements
5. **Customize**: Add custom metrics and plots as needed

## Reference

- **MetricsTracker**: Records and aggregates episode metrics
- **ComparisonPlotter**: Generates matplotlib visualizations
- **EpisodeMetrics**: Dataclass for individual episode data
- **AggregateMetrics**: Dataclass for aggregated statistics
