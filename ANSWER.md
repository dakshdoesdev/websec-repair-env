# Technical Training Q&A: WebSec Repair Gym

This document provides detailed answers to the technical checklist regarding the model training and dataset configuration.

---

### 1. Model & Format
- **What is the exact output format your model is trained to generate during fine-tuning?**
  The model is trained on a single concatenated string: `<instruction>\n<response>`.
  - **Instruction**: Includes system persona, task ID, code snippet, scanner hint, and JSON schema instructions.
  - **Response**: A nested JSON string: `{"action": {"action_type": "...", "vulnerability_type": "...", "patch_id": "..."}, "review": {...}}`.
- **How does your dataset represent each training example before tokenization?**
  As a Hugging Face `Dataset` object containing raw dictionary records with fields: `instruction`, `response`, `task_id`, `action_type`, `source`, and `example_family`.
- **Are you using a causal language model or a seq2seq model?**
  Causal language model (`AutoModelForCausalLM`).
- **Does the training loop expect flat JSON or nested JSON?**
  Nested JSON (`action.action_type`). It is parsed during inference by `normalize_action_payload` in `training/inference.py`.
- **How is the dual-JSON format (action + review) serialized into a string?**
  Using `json.dumps(payload, sort_keys=True)`.

---

### 2. Tokenization & Data Loading
- **What function in `training/data_loader.py` converts dataset examples into model-ready input_ids and labels?**
  None. `training/data_loader.py` handles loading and mixing datasets. The conversion to `input_ids` and `labels` is performed by the `preprocess_function` in `training/qlora_train.py`.
- **What tokenizer is being used and what padding/truncation settings are applied?**
  `AutoTokenizer` from `Qwen/Qwen2.5-Coder-1.5B-Instruct`.
  - `padding="max_length"`
  - `truncation=True`
  - `max_length=2048`
  - `padding_side="right"`
- **How are labels constructed from input_ids?**
  They are identical to `input_ids` (`tokenized["labels"] = tokenized["input_ids"].clone()`).
- **Are instruction and output separated during training?**
  No. They are concatenated, and no masking is applied to the instruction tokens (loss is computed on the full sequence).
- **What data collator is currently configured?**
  None explicitly (defaults to `DataCollatorWithPadding` via the `Trainer`, but labels are pre-generated).
- **Is `trust_remote_code=True` used?**
  Yes, for both the model and tokenizer.
- **Is `eos_token` used as `pad_token`?**
  Yes (`tokenizer.pad_token = tokenizer.eos_token`).

---

### 3. Hyperparameters & Hardware
- **What batch size and gradient accumulation settings are used?**
  - `per_device_train_batch_size=4`
  - `gradient_accumulation_steps=4`
- **What LoRA configuration is used?**
  - `rank (r)=16`, `alpha=32`, `target_modules=["q_proj", "v_proj"]`, `dropout=0.05`.
- **What quantization setup is used?**
  - `4-bit` (NF4), `double_quant=True`, `bnb_4bit_compute_dtype` is `bfloat16` or `float16`.
- **What learning rate and scheduler are configured?**
  - `2e-4`, `cosine` scheduler, `warmup_ratio=0.1`.
- **What GPU and VRAM are available?**
  - NVIDIA GeForce RTX 3050 Laptop GPU (~4.2 GB VRAM detected in logs).

---

### 4. Errors & Debugging
- **What exact error message and stack trace occurs at training crash?**
  The crash recorded in `websec_long_train.log` is:
  `RuntimeError: AirLLM teacher failed to initialize model 'Qwen/Qwen2.5-3B-Instruct': list index out of range`
- **At what line does the crash happen?**
  It happens in `training/teacher_backends.py` at line 210, triggered by `airllm/utils.py` at line 302 (`single_modelfile = shards[0]`) during teacher initialization. This is *before* the student QLoRA training starts.
- **Are there any malformed samples remaining?**
  `data_loader.py` includes normalization logic (`_normalize_training_output`), but the crash prevented training from reaching the sample validation stage.
- **Is there any OOM or CUDA-related warning?**
  No OOM was recorded, but the detected 4.2 GB VRAM is extremely low for training, making OOM likely if the 3B model were successfully initialized.

---

### 5. Inference & Evaluation
- **Does the inference script correctly parse nested JSON?**
  Yes, `training/inference.py` uses `extract_json_object` and `normalize_action_payload` to handle nested structures.
- **Is the inference prompt same as training prompt?**
  No. Training uses a concatenated instruction/response, while inference uses a more verbose template built by `_build_prompt` in `WebSecRepairAgent`.
- **Does the dataset include negative examples (safe code)?**
  Yes, through the integration of SWE-bench Lite and potential manual entries in `websec_repair_train_100.jsonl`.
- **Are metrics being computed during training?**
  Yes, `eval_loss` is tracked using a 10% validation split.
