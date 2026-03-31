#!/usr/bin/env python3
"""Generate comparison visualization from real training runs."""

import json
from visualization import MetricsTracker, ComparisonPlotter


def load_tracker_from_json(filepath: str, run_name: str) -> MetricsTracker:
    """Load tracker from saved JSON."""
    with open(filepath) as f:
        data = json.load(f)
    
    tracker = MetricsTracker(run_name=run_name)
    for ep_data in data['episodes']:
        tracker.add_episode(
            episode=ep_data['episode'],
            task_id=ep_data['task_id'],
            cumulative_reward=ep_data['cumulative_reward'],
            success=ep_data['success'],
            steps=ep_data['steps'],
            vulnerability_correct=ep_data['vulnerability_correct'],
            patch_correct=ep_data['patch_correct'],
            time_seconds=ep_data['time_seconds'],
        )
    
    return tracker


def main():
    print("📊 GENERATING COMPARISON REPORT")
    print("=" * 70)
    
    # Load trackers
    print("\n✓ Loading baseline metrics...")
    base_tracker = load_tracker_from_json("baseline_run/metrics.json", "baseline_agent")
    base_agg = base_tracker.get_aggregate()
    
    print("✓ Loading trained metrics...")
    trained_tracker = load_tracker_from_json("trained_run/metrics.json", "trained_agent")
    trained_agg = trained_tracker.get_aggregate()
    
    # Display metrics
    print("\n" + "=" * 70)
    print("📈 BASELINE AGENT")
    print("=" * 70)
    print(f"Success Rate:         {base_agg.success_pct:>6.1f}%")
    print(f"Success Count:        {base_agg.success_count}/{base_agg.total_episodes}")
    print(f"Avg Reward:           {base_agg.avg_reward:>6.3f}")
    print(f"Avg Steps:            {base_agg.avg_steps:>6}")
    print(f"Vulnerability Acc:    {base_agg.vulnerability_accuracy*100:>6.1f}%")
    print(f"Patch Selection Acc:  {base_agg.patch_accuracy*100:>6.1f}%")
    
    print("\n" + "=" * 70)
    print("📈 TRAINED AGENT")
    print("=" * 70)
    print(f"Success Rate:         {trained_agg.success_pct:>6.1f}%")
    print(f"Success Count:        {trained_agg.success_count}/{trained_agg.total_episodes}")
    print(f"Avg Reward:           {trained_agg.avg_reward:>6.3f}")
    print(f"Avg Steps:            {trained_agg.avg_steps:>6}")
    print(f"Vulnerability Acc:    {trained_agg.vulnerability_accuracy*100:>6.1f}%")
    print(f"Patch Selection Acc:  {trained_agg.patch_accuracy*100:>6.1f}%")
    
    # Calculate improvements
    print("\n" + "=" * 70)
    print("🎯 IMPROVEMENTS")
    print("=" * 70)
    success_imp = trained_agg.success_pct - base_agg.success_pct
    reward_imp = trained_agg.avg_reward - base_agg.avg_reward
    steps_imp = base_agg.avg_steps - trained_agg.avg_steps
    vuln_imp = trained_agg.vulnerability_accuracy - base_agg.vulnerability_accuracy
    patch_imp = trained_agg.patch_accuracy - base_agg.patch_accuracy
    
    print(f"Success Rate:         {success_imp:>+6.1f}%")
    print(f"Avg Reward:           {reward_imp:>+6.3f}")
    print(f"Avg Steps:            {steps_imp:>+6} (lower is better)")
    print(f"Vulnerability Acc:    {vuln_imp*100:>+6.1f}%")
    print(f"Patch Selection Acc:  {patch_imp*100:>+6.1f}%")
    
    # Generate visualizations
    print("\n" + "=" * 70)
    print("🎨 GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    plotter = ComparisonPlotter(output_dir="training_comparison_results")
    
    print("✓ Success rate comparison...")
    plotter.plot_success_rate_comparison(base_tracker, trained_tracker)
    
    print("✓ Reward trajectory...")
    plotter.plot_reward_trajectory(base_tracker, trained_tracker)
    
    print("✓ Average metrics...")
    plotter.plot_average_metrics(base_tracker, trained_tracker)
    
    print("✓ Accuracy metrics...")
    plotter.plot_accuracy_comparison(base_tracker, trained_tracker)
    
    print("✓ HTML report...")
    report_path = plotter.generate_html_report(
        base_tracker,
        trained_tracker,
        report_title="WebSec Repair Environment - Training Results"
    )
    
    print("\n" + "=" * 70)
    print("✅ COMPARISON COMPLETE!")
    print("=" * 70)
    print(f"\n📂 Output directory: training_comparison_results/")
    print(f"📄 Report:          {report_path}")
    print("\nGenerated files:")
    print("  ├── report.html")
    print("  ├── 01_success_rate.png")
    print("  ├── 02_reward_trajectory.png")
    print("  ├── 03_average_metrics.png")
    print("  └── 04_accuracy_comparison.png")
    print("\n👉 Open report.html in your browser to view the comparison!")


if __name__ == "__main__":
    main()
