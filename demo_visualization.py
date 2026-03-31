#!/usr/bin/env python3
"""Demo script showing visualization capabilities with sample data."""

from visualization import MetricsTracker, ComparisonPlotter


def create_sample_base_metrics() -> MetricsTracker:
    """Create sample metrics for base (untrained) model."""
    tracker = MetricsTracker(run_name="base_model")
    
    # Simulate base model performance - low success rate, low rewards
    for episode in range(1, 11):
        success = episode > 7  # Only last 2 succeed (20% success rate)
        reward = 0.1 + (0.05 if success else 0)
        steps = 5 if success else 6
        
        tracker.add_episode(
            episode=episode,
            task_id="sqli_login",
            cumulative_reward=reward,
            success=success,
            steps=steps,
            vulnerability_correct=(success or episode % 2 == 0),
            patch_correct=success,
            time_seconds=2.5 + (1.0 if success else 0),
        )
    
    return tracker


def create_sample_trained_metrics() -> MetricsTracker:
    """Create sample metrics for trained model."""
    tracker = MetricsTracker(run_name="trained_model")
    
    # Simulate trained model performance - high success rate, high rewards
    for episode in range(1, 11):
        success = episode > 3  # 70% success rate
        reward = 0.3 + (0.4 if success else 0.1)
        steps = 4 if success else 5
        
        tracker.add_episode(
            episode=episode,
            task_id="sqli_login",
            cumulative_reward=reward,
            success=success,
            steps=steps,
            vulnerability_correct=(episode > 2),
            patch_correct=(episode > 2),
            time_seconds=2.0 + (0.5 if success else 0),
        )
    
    return tracker


def main():
    """Generate visualization report."""
    print("🎯 WebSec Repair Environment - Visualization Demo")
    print("=" * 60)
    
    # Create sample metrics
    print("\n📊 Creating sample metrics...")
    base_tracker = create_sample_base_metrics()
    trained_tracker = create_sample_trained_metrics()
    
    # Save metrics to JSON
    print("💾 Saving metrics to JSON...")
    base_tracker.save_json("visualization_results/base_metrics.json")
    trained_tracker.save_json("visualization_results/trained_metrics.json")
    
    # Display aggregate metrics
    base_agg = base_tracker.get_aggregate()
    trained_agg = trained_tracker.get_aggregate()
    
    print("\n📈 BASE MODEL METRICS:")
    print(f"  Success Rate: {base_agg.success_pct:.1f}%")
    print(f"  Success Count: {base_agg.success_count}/{base_agg.total_episodes}")
    print(f"  Avg Reward: {base_agg.avg_reward:.3f}")
    print(f"  Avg Steps: {base_agg.avg_steps}")
    print(f"  Vulnerability Accuracy: {base_agg.vulnerability_accuracy*100:.1f}%")
    print(f"  Patch Accuracy: {base_agg.patch_accuracy*100:.1f}%")
    
    print("\n📈 TRAINED MODEL METRICS:")
    print(f"  Success Rate: {trained_agg.success_pct:.1f}%")
    print(f"  Success Count: {trained_agg.success_count}/{trained_agg.total_episodes}")
    print(f"  Avg Reward: {trained_agg.avg_reward:.3f}")
    print(f"  Avg Steps: {trained_agg.avg_steps}")
    print(f"  Vulnerability Accuracy: {trained_agg.vulnerability_accuracy*100:.1f}%")
    print(f"  Patch Accuracy: {trained_agg.patch_accuracy*100:.1f}%")
    
    print("\n📊 IMPROVEMENT:")
    improvement_pct = trained_agg.success_pct - base_agg.success_pct
    improvement_reward = trained_agg.avg_reward - base_agg.avg_reward
    print(f"  Success Rate: {improvement_pct:+.1f}%")
    print(f"  Avg Reward: {improvement_reward:+.3f}")
    
    # Generate plots
    print("\n🎨 Generating visualization plots...")
    plotter = ComparisonPlotter(output_dir="visualization_results")
    
    print("  ✓ Success rate comparison")
    plotter.plot_success_rate_comparison(base_tracker, trained_tracker)
    
    print("  ✓ Reward trajectory")
    plotter.plot_reward_trajectory(base_tracker, trained_tracker)
    
    print("  ✓ Average metrics")
    plotter.plot_average_metrics(base_tracker, trained_tracker)
    
    print("  ✓ Accuracy comparison")
    plotter.plot_accuracy_comparison(base_tracker, trained_tracker)
    
    # Generate HTML report
    print("\n📄 Generating HTML report...")
    report_path = plotter.generate_html_report(
        base_tracker,
        trained_tracker,
        report_title="WebSec Repair Environment - Training Comparison"
    )
    
    print("\n" + "=" * 60)
    print("✅ VISUALIZATION COMPLETE!")
    print(f"📂 Output directory: visualization_results/")
    print(f"📄 Report: {report_path}")
    print("\nGenerated files:")
    print("  - report.html (main report with all metrics and plots)")
    print("  - 01_success_rate.png")
    print("  - 02_reward_trajectory.png")
    print("  - 03_average_metrics.png")
    print("  - 04_accuracy_comparison.png")
    print("  - base_metrics.json")
    print("  - trained_metrics.json")
    print("\n👉 Open report.html in your browser to view the full report!")


if __name__ == "__main__":
    main()
