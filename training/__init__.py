"""WebSec Repair training module.

Contains:
- data_loader: Dataset loading and preprocessing
- qlora_train: QLoRA fine-tuning pipeline
- inference: Inference with trained models
"""

from __future__ import annotations

try:
    from training.data_loader import (
        create_combined_dataset,
        create_report_training_data,
        create_synthetic_training_data,
        create_training_dataset,
        load_jsonl_dataset,
        load_trajectory_metrics,
        load_trajectory_logs,
        summarize_dataset,
    )
    from training.build_external_websec_corpus import build_external_websec_corpus
    from training.inference import WebSecRepairAgent
except ImportError:
    # Graceful fallback when imports fail
    pass

__all__ = [
    "create_combined_dataset",
    "create_report_training_data",
    "create_synthetic_training_data",
    "create_training_dataset",
    "load_jsonl_dataset",
    "load_trajectory_metrics",
    "load_trajectory_logs",
    "summarize_dataset",
    "build_external_websec_corpus",
    "WebSecRepairAgent",
]
