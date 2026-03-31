#!/usr/bin/env python3
"""QLoRA training script for Qwen2.5-Coder-1.5B on WebSec repair tasks.

This script is designed to run on Google Colab with free GPU.
It trains a model to identify and classify vulnerabilities, then apply patches.

To use in Colab:
1. Upload this script to Colab
2. Run: pip install -r training_requirements.txt
3. Run: python training/qlora_train.py --output-dir ./qwen-websec-finetuned

Optional: Save to Hugging Face Hub with --hf-model-id your-username/qwen-websec
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.data_loader import create_combined_dataset, summarize_dataset


def create_prompt(example: dict) -> str:
    """Format example into a single prompt string."""
    instruction = example.get("instruction", "")
    response = example.get("response", "")
    return f"{instruction}\n{response}"


def preprocess_function(examples: dict, tokenizer) -> dict:
    """Tokenize instructions and responses."""
    prompts = [
        create_prompt({"instruction": inst, "response": resp})
        for inst, resp in zip(examples["instruction"], examples["response"])
    ]
    
    tokenized = tokenizer(
        prompts,
        max_length=2048,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    # Labels are same as input_ids for language modeling loss
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized


def _resolve_default_data_paths(
    jsonl_dataset_path: str | None,
    trajectory_logs_path: str | None,
    research_doc_path: str | None,
    external_websec_dataset_path: str | None,
) -> tuple[str | None, str | None, str | None, str | None]:
    repo_root = REPO_ROOT

    resolved_jsonl = jsonl_dataset_path
    if resolved_jsonl is None:
        default_jsonl = repo_root / "training" / "websec_repair_train_100.jsonl"
        if default_jsonl.exists():
            resolved_jsonl = str(default_jsonl)

    resolved_trajectory = trajectory_logs_path
    if resolved_trajectory is None:
        default_trajectory = repo_root / "trajectory_logs.json"
        if default_trajectory.exists():
            resolved_trajectory = str(default_trajectory)

    resolved_research = research_doc_path
    if resolved_research is None:
        default_research = repo_root / "training" / "research_websec_review_playbook.md"
        if default_research.exists():
            resolved_research = str(default_research)

    resolved_external = external_websec_dataset_path
    if resolved_external is None:
        default_external = repo_root / "training" / "external_websec_corpus.jsonl"
        if default_external.exists():
            resolved_external = str(default_external)

    return resolved_jsonl, resolved_trajectory, resolved_research, resolved_external


def _build_training_arguments(
    *,
    output_dir: str,
    num_train_epochs: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    train_examples: int,
) -> Any:
    """Create TrainingArguments with compatibility across transformers versions."""

    import torch
    from transformers import TrainingArguments

    supports = inspect.signature(TrainingArguments.__init__).parameters
    cuda_available = torch.cuda.is_available()
    use_bf16 = bool(cuda_available and torch.cuda.is_bf16_supported())
    use_fp16 = bool(cuda_available and not use_bf16)
    effective_batch_size = max(1, per_device_batch_size * gradient_accumulation_steps)
    steps_per_epoch = max(1, math.ceil(train_examples / effective_batch_size))
    logging_steps = max(1, min(10, steps_per_epoch))
    eval_save_steps = max(10, steps_per_epoch)

    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_batch_size,
        "per_device_eval_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "optim": "paged_adamw_8bit",
        "logging_steps": logging_steps,
        "save_steps": eval_save_steps,
        "eval_steps": eval_save_steps,
        "save_strategy": "steps",
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": [],
    }

    if "overwrite_output_dir" in supports:
        kwargs["overwrite_output_dir"] = True
    if "evaluation_strategy" in supports:
        kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in supports:
        kwargs["eval_strategy"] = "steps"
    if "bf16" in supports:
        kwargs["bf16"] = use_bf16
    if "fp16" in supports:
        kwargs["fp16"] = use_fp16

    return TrainingArguments(**kwargs)


def _print_counter(label: str, counts: dict[str, int]) -> None:
    print(f"   {label}:")
    if not counts:
        print("     - none")
        return
    for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"     - {key}: {value}")


def _print_dataset_summary(summary: dict[str, Any], train_size: int, eval_size: int) -> None:
    print(f"   Total examples: {summary['total_examples']}")
    _print_counter("Source breakdown", summary["source_breakdown"])
    _print_counter("Action type counts", summary["action_type_counts"])
    _print_counter("Vulnerability counts", summary["vulnerability_type_counts"])
    print(f"   Train split: {train_size}")
    print(f"   Eval split: {eval_size}")


def train_qwen_websec(
    output_dir: str = "./qwen-websec-finetuned",
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    hf_model_id: str | None = None,
    num_train_epochs: int = 3,
    learning_rate: float = 2e-4,
    per_device_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    jsonl_dataset_path: str | None = None,
    trajectory_logs_path: str | None = None,
    research_doc_path: str | None = None,
    external_websec_dataset_path: str | None = None,
    include_swe_bench: bool = False,
    include_synthetic: bool = True,
    allow_tiny_debug_dataset: bool = False,
) -> None:
    """Train Qwen2.5-Coder-1.5B with QLoRA on WebSec repair data.
    
    Args:
        output_dir: Directory to save the fine-tuned model
        model_name: Hugging Face model ID
        hf_model_id: Optional HF Hub ID to push to (e.g., "username/qwen-websec")
        num_train_epochs: Number of training epochs
        learning_rate: Learning rate
        per_device_batch_size: Batch size per GPU
        gradient_accumulation_steps: Steps to accumulate gradients
        jsonl_dataset_path: Path to the primary WebSec JSONL dataset
        trajectory_logs_path: Path to trajectory_logs.json from generate_trajectories.py
        research_doc_path: Path to a markdown/txt security research writeup
        external_websec_dataset_path: Path to a normalized external WebSec corpus JSONL
        include_swe_bench: Whether to include SWE-bench Lite (opt-in only)
        include_synthetic: Whether to include synthetic examples (6 examples)
        allow_tiny_debug_dataset: Allow training on fewer than 50 examples
    """
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
    )

    print("=" * 80)
    print("WebSec Repair - QLoRA Training")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {num_train_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {per_device_batch_size}")
    print()
    
    # Check GPU
    if not torch.cuda.is_available():
        print("WARNING: No CUDA GPU detected. Training will be very slow on CPU.")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print()
    
    # 1. Load and prepare dataset
    print("1. Loading training data...")
    (
        jsonl_dataset_path,
        trajectory_logs_path,
        research_doc_path,
        external_websec_dataset_path,
    ) = _resolve_default_data_paths(
        jsonl_dataset_path,
        trajectory_logs_path,
        research_doc_path,
        external_websec_dataset_path,
    )

    if jsonl_dataset_path:
        print(f"   JSONL dataset: {jsonl_dataset_path}")
    if trajectory_logs_path:
        print(f"   Trajectory logs: {trajectory_logs_path}")
    if research_doc_path:
        print(f"   Research doc: {research_doc_path}")
    if external_websec_dataset_path:
        print(f"   External WebSec corpus: {external_websec_dataset_path}")

    full_dataset = create_combined_dataset(
        jsonl_dataset_path=jsonl_dataset_path,
        trajectory_logs_path=trajectory_logs_path,
        research_doc_path=research_doc_path,
        external_websec_dataset_path=external_websec_dataset_path,
        include_swe_bench=include_swe_bench,
        include_synthetic=include_synthetic,
    )

    total_examples = len(full_dataset)
    if total_examples < 50 and not allow_tiny_debug_dataset:
        raise ValueError(
            "Refusing to train on fewer than 50 examples. "
            "Pass --allow-tiny-debug-dataset only for loader debugging."
        )
    if total_examples < 50:
        print("   WARNING: tiny debug dataset mode enabled.")

    # Split into train/eval
    if total_examples > 1:
        test_size = max(1, round(total_examples * 0.1))
        split = full_dataset.train_test_split(test_size=test_size, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = full_dataset
        eval_dataset = Dataset.from_dict({column: [] for column in full_dataset.column_names})

    dataset_summary = summarize_dataset(full_dataset)
    _print_dataset_summary(dataset_summary, len(train_dataset), len(eval_dataset))
    print()
    
    # 2. Load tokenizer and model
    print("2. Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    print(f"   Tokenizer loaded (vocab size: {len(tokenizer)})")
    
    # BitsAndBytes config for 4-bit quantization
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"   Model loaded (params: {model.num_parameters():,})")
    model = prepare_model_for_kbit_training(model)
    print()
    
    # 3. Configure LoRA
    print("3. Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable params: {trainable:,}")
    print(f"   Total params: {total:,}")
    print(f"   Trainable ratio: {100 * trainable / total:.2f}%")
    print()
    
    # 4. Tokenize dataset
    print("4. Tokenizing dataset...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names,
    )
    print(f"   Tokenization complete")
    print()
    
    # 5. Training arguments
    print("5. Setting up training...")
    training_args = _build_training_arguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        train_examples=len(train_dataset),
    )
    
    # 6. Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=None,  # Let the trainer handle it
    )
    
    # 7. Train!
    print("6. Starting training...")
    print()
    trainer.train()
    
    # 8. Save model
    print()
    print("7. Saving model...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(output_path / "adapter")
    tokenizer.save_pretrained(output_path / "tokenizer")
    
    # Save training info
    training_info = {
        "model_name": model_name,
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.05,
        },
        "training_args": {
            "num_train_epochs": num_train_epochs,
            "learning_rate": learning_rate,
            "per_device_batch_size": per_device_batch_size,
        },
        "final_loss": float(trainer.state.best_metric or 0),
    }
    
    with open(output_path / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print(f"✓ Model saved to {output_dir}")
    print(f"  - LoRA adapter: {output_path}/adapter")
    print(f"  - Tokenizer: {output_path}/tokenizer")
    print(f"  - Info: {output_path}/training_info.json")
    print()
    
    # 9. Push to Hugging Face (optional)
    if hf_model_id:
        print(f"8. Pushing to Hugging Face Hub ({hf_model_id})...")
        try:
            model.push_to_hub(hf_model_id, token=os.getenv("HF_TOKEN"))
            print(f"✓ Pushed to https://huggingface.co/{hf_model_id}")
        except Exception as e:
            print(f"✗ Failed to push to Hub: {e}")
        print()
    
    print("=" * 80)
    print("Training complete!")
    print("=" * 80)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train Qwen2.5-Coder for WebSec repair")
    parser.add_argument(
        "--output-dir",
        default="./qwen-websec-finetuned",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="Hugging Face model ID to train",
    )
    parser.add_argument(
        "--hf-model-id",
        default=None,
        help="Hugging Face Hub model ID to push to (e.g., username/qwen-websec)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size per device",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--jsonl-dataset",
        type=str,
        default=None,
        help="Path to the primary WebSec JSONL dataset",
    )
    parser.add_argument(
        "--trajectory-logs",
        type=str,
        default=None,
        help="Path to trajectory_logs.json from generate_trajectories.py",
    )
    parser.add_argument(
        "--research-doc",
        type=str,
        default=None,
        help="Path to a markdown/txt security research writeup used for report-derived SFT examples",
    )
    parser.add_argument(
        "--external-websec-jsonl",
        type=str,
        default=None,
        help="Path to a normalized external WebSec JSONL corpus built from PrimeVul/Devign/Juliet/OWASP/patch sources",
    )
    parser.add_argument(
        "--include-swe-bench",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to include SWE-bench Lite data (default: False)",
    )
    parser.add_argument(
        "--include-synthetic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include synthetic data (default: True)",
    )
    parser.add_argument(
        "--allow-tiny-debug-dataset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow training on fewer than 50 examples for loader debugging",
    )
    
    args = parser.parse_args()
    
    train_qwen_websec(
        output_dir=args.output_dir,
        model_name=args.model_id,
        hf_model_id=args.hf_model_id,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        jsonl_dataset_path=args.jsonl_dataset,
        trajectory_logs_path=args.trajectory_logs,
        research_doc_path=args.research_doc,
        external_websec_dataset_path=args.external_websec_jsonl,
        include_swe_bench=args.include_swe_bench,
        include_synthetic=args.include_synthetic,
        allow_tiny_debug_dataset=args.allow_tiny_debug_dataset,
    )


if __name__ == "__main__":
    main()
