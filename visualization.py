#!/usr/bin/env python3
"""Visualization and metrics tracking for WebSec Repair Environment training."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use('Agg')  # Headless mode for servers
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    episode: int
    task_id: str
    cumulative_reward: float
    success: bool
    steps: int
    vulnerability_correct: bool
    patch_correct: bool
    time_seconds: float


@dataclass
class AggregateMetrics:
    """Aggregate metrics across episodes."""
    total_episodes: int
    success_count: int
    success_rate: float
    avg_reward: float
    avg_steps: int
    vulnerability_accuracy: float
    patch_accuracy: float
    
    @property
    def success_pct(self) -> float:
        """Success rate as percentage."""
        return self.success_rate * 100


class MetricsTracker:
    """Track metrics during agent training/inference."""
    
    def __init__(self, run_name: str = "run"):
        self.run_name = run_name
        self.episodes: list[EpisodeMetrics] = []
        self.start_time: float | None = None
        
    def add_episode(
        self,
        episode: int,
        task_id: str,
        cumulative_reward: float,
        success: bool,
        steps: int,
        vulnerability_correct: bool = False,
        patch_correct: bool = False,
        time_seconds: float = 0.0,
    ) -> None:
        """Record metrics for an episode."""
        metrics = EpisodeMetrics(
            episode=episode,
            task_id=task_id,
            cumulative_reward=cumulative_reward,
            success=success,
            steps=steps,
            vulnerability_correct=vulnerability_correct,
            patch_correct=patch_correct,
            time_seconds=time_seconds,
        )
        self.episodes.append(metrics)
    
    def get_aggregate(self) -> AggregateMetrics:
        """Calculate aggregate metrics."""
        if not self.episodes:
            return AggregateMetrics(0, 0, 0.0, 0.0, 0, 0.0, 0.0)
        
        success_count = sum(1 for e in self.episodes if e.success)
        success_rate = success_count / len(self.episodes)
        avg_reward = np.mean([e.cumulative_reward for e in self.episodes])
        avg_steps = int(np.mean([e.steps for e in self.episodes]))
        
        vuln_correct = sum(1 for e in self.episodes if e.vulnerability_correct)
        patch_correct = sum(1 for e in self.episodes if e.patch_correct)
        
        vuln_accuracy = vuln_correct / len(self.episodes) if self.episodes else 0.0
        patch_accuracy = patch_correct / len(self.episodes) if self.episodes else 0.0
        
        return AggregateMetrics(
            total_episodes=len(self.episodes),
            success_count=success_count,
            success_rate=success_rate,
            avg_reward=avg_reward,
            avg_steps=avg_steps,
            vulnerability_accuracy=vuln_accuracy,
            patch_accuracy=patch_accuracy,
        )
    
    def save_json(self, filepath: str | Path) -> None:
        """Save metrics to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "run_name": self.run_name,
            "episodes": [asdict(e) for e in self.episodes],
            "aggregate": asdict(self.get_aggregate()),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load_json(filepath: str | Path) -> dict:
        """Load metrics from JSON file."""
        with open(filepath) as f:
            return json.load(f)


class ComparisonPlotter:
    """Generate comparison plots for base vs. trained models."""
    
    def __init__(self, output_dir: str | Path = "visualization_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_success_rate_comparison(
        self,
        base_tracker: MetricsTracker,
        trained_tracker: MetricsTracker,
        title: str = "Success Rate: Base vs. Trained",
    ) -> str:
        """Plot success rate comparison."""
        base_agg = base_tracker.get_aggregate()
        trained_agg = trained_tracker.get_aggregate()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        models = ['Base Model', 'Trained Model']
        rates = [base_agg.success_pct, trained_agg.success_pct]
        colors = ['#ef4444', '#10b981']
        
        bars = ax.bar(models, rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add percentage labels on bars
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        
        filepath = self.output_dir / "01_success_rate.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_reward_trajectory(
        self,
        base_tracker: MetricsTracker,
        trained_tracker: MetricsTracker,
        title: str = "Reward Trajectory Over Episodes",
    ) -> str:
        """Plot reward progression over episodes."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if base_tracker.episodes:
            episodes_b = [e.episode for e in base_tracker.episodes]
            rewards_b = [e.cumulative_reward for e in base_tracker.episodes]
            ax.plot(episodes_b, rewards_b, 'o-', label='Base Model', 
                   color='#ef4444', linewidth=2, markersize=6, alpha=0.7)
        
        if trained_tracker.episodes:
            episodes_t = [e.episode for e in trained_tracker.episodes]
            rewards_t = [e.cumulative_reward for e in trained_tracker.episodes]
            ax.plot(episodes_t, rewards_t, 's-', label='Trained Model',
                   color='#10b981', linewidth=2, markersize=6, alpha=0.7)
        
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Reward', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        filepath = self.output_dir / "02_reward_trajectory.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_average_metrics(
        self,
        base_tracker: MetricsTracker,
        trained_tracker: MetricsTracker,
        title: str = "Average Metrics Comparison",
    ) -> str:
        """Plot average reward and steps comparison."""
        base_agg = base_tracker.get_aggregate()
        trained_agg = trained_tracker.get_aggregate()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Average reward
        models = ['Base', 'Trained']
        rewards = [base_agg.avg_reward, trained_agg.avg_reward]
        colors = ['#ef4444', '#10b981']
        
        bars1 = ax1.bar(models, rewards, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        for bar, reward in zip(bars1, rewards):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{reward:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax1.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
        ax1.set_title('Avg Reward per Episode', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Average steps
        steps = [base_agg.avg_steps, trained_agg.avg_steps]
        bars2 = ax2.bar(models, steps, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        for bar, step in zip(bars2, steps):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax2.set_ylabel('Average Steps', fontsize=12, fontweight='bold')
        ax2.set_title('Avg Steps to Solution', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        filepath = self.output_dir / "03_average_metrics.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_accuracy_comparison(
        self,
        base_tracker: MetricsTracker,
        trained_tracker: MetricsTracker,
        title: str = "Classification & Patch Accuracy",
    ) -> str:
        """Plot vulnerability and patch accuracy comparison."""
        base_agg = base_tracker.get_aggregate()
        trained_agg = trained_tracker.get_aggregate()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(2)
        width = 0.35
        
        vuln_accs = [base_agg.vulnerability_accuracy * 100, trained_agg.vulnerability_accuracy * 100]
        patch_accs = [base_agg.patch_accuracy * 100, trained_agg.patch_accuracy * 100]
        
        bars1 = ax.bar(x - width/2, vuln_accs, width, label='Vulnerability Accuracy',
                      color='#3b82f6', alpha=0.7, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, patch_accs, width, label='Patch Accuracy',
                      color='#f59e0b', alpha=0.7, edgecolor='black', linewidth=1.5)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Base Model', 'Trained Model'])
        ax.legend(fontsize=11)
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3)
        
        filepath = self.output_dir / "04_accuracy_comparison.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def generate_html_report(
        self,
        base_tracker: MetricsTracker,
        trained_tracker: MetricsTracker,
        report_title: str = "Training Comparison Report",
    ) -> str:
        """Generate HTML report with all plots and metrics."""
        # Generate all plots
        plot1 = self.plot_success_rate_comparison(base_tracker, trained_tracker)
        plot2 = self.plot_reward_trajectory(base_tracker, trained_tracker)
        plot3 = self.plot_average_metrics(base_tracker, trained_tracker)
        plot4 = self.plot_accuracy_comparison(base_tracker, trained_tracker)
        
        base_agg = base_tracker.get_aggregate()
        trained_agg = trained_tracker.get_aggregate()
        
        improvement = trained_agg.success_pct - base_agg.success_pct
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_title}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%); padding: 30px; border-radius: 12px; margin-bottom: 30px; border-left: 4px solid #2563eb; }}
        .header h1 {{ font-size: 2.2rem; font-weight: 700; color: #60a5fa; margin-bottom: 10px; }}
        .header p {{ color: #94a3b8; font-size: 1.1rem; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .metric-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; }}
        .metric-card h3 {{ color: #93c5fd; margin-bottom: 10px; font-size: 0.9rem; text-transform: uppercase; }}
        .metric-value {{ font-size: 2rem; font-weight: 700; color: #10b981; margin-bottom: 5px; }}
        .metric-label {{ color: #94a3b8; font-size: 0.9rem; }}
        .improvement {{ color: #f59e0b; font-weight: 600; }}
        .plots-section {{ margin-top: 40px; }}
        .plot-container {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; margin-bottom: 30px; }}
        .plot-container h2 {{ color: #93c5fd; margin-bottom: 15px; font-size: 1.3rem; }}
        .plot-container img {{ max-width: 100%; height: auto; border-radius: 8px; }}
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }}
        .summary {{ background: #1e293b; border: 2px solid #2563eb; border-radius: 12px; padding: 25px; margin-bottom: 30px; }}
        .summary h2 {{ color: #60a5fa; margin-bottom: 15px; }}
        .summary p {{ color: #cbd5e1; margin-bottom: 10px; line-height: 1.6; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 {report_title}</h1>
            <p>Side-by-side comparison of base model vs. trained model performance</p>
        </div>
        
        <div class="summary">
            <h2>🎯 Key Findings</h2>
            <p><strong>Success Rate Improvement:</strong> <span class="improvement">{improvement:+.1f}%</span> 
               ({base_agg.success_pct:.1f}% → {trained_agg.success_pct:.1f}%)</p>
            <p><strong>Reward Improvement:</strong> <span class="improvement">{trained_agg.avg_reward - base_agg.avg_reward:+.2f}</span> 
               ({base_agg.avg_reward:.2f} → {trained_agg.avg_reward:.2f})</p>
            <p><strong>Efficiency:</strong> Base model took {base_agg.avg_steps} steps on average, 
               trained model took {trained_agg.avg_steps} steps.</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Base Model</h3>
                <div class="metric-value">{base_agg.success_pct:.1f}%</div>
                <div class="metric-label">Success Rate</div>
                <div style="margin-top: 15px; font-size: 0.9rem; color: #cbd5e1;">
                    {base_agg.success_count}/{base_agg.total_episodes} solved
                </div>
            </div>
            
            <div class="metric-card">
                <h3>Trained Model</h3>
                <div class="metric-value">{trained_agg.success_pct:.1f}%</div>
                <div class="metric-label">Success Rate</div>
                <div style="margin-top: 15px; font-size: 0.9rem; color: #cbd5e1;">
                    {trained_agg.success_count}/{trained_agg.total_episodes} solved
                </div>
            </div>
            
            <div class="metric-card">
                <h3>Base Model</h3>
                <div class="metric-value">{base_agg.avg_reward:.2f}</div>
                <div class="metric-label">Avg Reward</div>
            </div>
            
            <div class="metric-card">
                <h3>Trained Model</h3>
                <div class="metric-value">{trained_agg.avg_reward:.2f}</div>
                <div class="metric-label">Avg Reward</div>
            </div>
        </div>
        
        <div class="plots-section">
            <div class="plot-container">
                <h2>Success Rate Comparison</h2>
                <img src="{Path(plot1).name}" alt="Success Rate Comparison">
            </div>
            
            <div class="plot-container">
                <h2>Reward Progression</h2>
                <img src="{Path(plot2).name}" alt="Reward Trajectory">
            </div>
            
            <div class="grid-2">
                <div class="plot-container">
                    <h2>Average Metrics</h2>
                    <img src="{Path(plot3).name}" alt="Average Metrics">
                </div>
                
                <div class="plot-container">
                    <h2>Accuracy Metrics</h2>
                    <img src="{Path(plot4).name}" alt="Accuracy Comparison">
                </div>
            </div>
        </div>
        
        <div style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #334155;">
            <p style="color: #64748b; font-size: 0.9rem;">Generated by WebSec Repair Environment Visualization System</p>
        </div>
    </div>
</body>
</html>
"""
        
        report_path = self.output_dir / "report.html"
        with open(report_path, 'w') as f:
            f.write(html)
        
        return str(report_path)
