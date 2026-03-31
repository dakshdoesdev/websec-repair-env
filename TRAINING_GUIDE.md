# Training Guide: Qwen2.5-Coder for WebSec Repair

This guide walks you through fine-tuning **Qwen2.5-Coder-1.5B-Instruct** on your WebSec vulnerability repair tasks using Google Colab with free GPU.

## Quick Start (5 minutes)

### 1. Copy to Colab

```bash
# In Colab, clone the repo
!git clone https://github.com/dakshdoesdev/imkc.git
%cd imkc
```

### 2. Install Dependencies

```bash
# Install training requirements
!pip install -q -r training_requirements.txt
```

### 3. Run Training

```bash
# Optional: build a normalized external WebSec corpus if you exported PrimeVul,
# Devign, Juliet, OWASP Benchmark, Big-Vul, CVEfixes, or GitHub security fixes.
!python training/build_external_websec_corpus.py \
  --output training/external_websec_corpus.jsonl \
  --primevul /content/data/primevul.jsonl \
  --devign /content/data/devign.json \
  --juliet /content/data/juliet \
  --owasp-benchmark /content/data/owasp_benchmark.jsonl \
  --bigvul /content/data/bigvul.csv \
  --github-security-fixes /content/data/github_security_fixes.jsonl

# Start fine-tuning (will take ~15-30 minutes depending on GPU)
!python training/qlora_train.py \
  --output-dir ./qwen-websec-finetuned \
  --jsonl-dataset training/websec_repair_train_100.jsonl \
  --trajectory-logs trajectory_logs.json \
  --research-doc training/research_websec_review_playbook.md \
  --external-websec-jsonl training/external_websec_corpus.jsonl \
  --no-include-swe-bench \
  --epochs 3 \
  --batch-size 4
```

### 4. Done!

Your fine-tuned model is in `./qwen-websec-finetuned/`

---

## Detailed Setup

### Prerequisites

- Google Colab account (free with Gmail)
- 15GB+ free storage (for model weights)
- Hugging Face account (optional, for saving to Hub)

### Full Colab Notebook Steps

#### Cell 1: Clone repo and install dependencies

```python
# Clone the WebSec environment
!git clone https://github.com/dakshdoesdev/imkc.git
%cd imkc

# Install core dependencies
!pip install -q -r requirements.txt
!pip install -q -r training_requirements.txt

# Verify GPU
!nvidia-smi
```

Expected output: Shows your GPU (usually A100 or T4)

#### Cell 2: Run training

```python
!python training/build_external_websec_corpus.py \
  --output training/external_websec_corpus.jsonl \
  --primevul /content/data/primevul.jsonl \
  --devign /content/data/devign.json \
  --juliet /content/data/juliet \
  --owasp-benchmark /content/data/owasp_benchmark.jsonl \
  --bigvul /content/data/bigvul.csv \
  --cvefixes /content/data/cvefixes.jsonl \
  --github-security-fixes /content/data/github_security_fixes.jsonl \
  --nvd-cve /content/data/nvd_curated.jsonl \
  --exploitdb-selected /content/data/exploitdb_selected.jsonl

!python training/qlora_train.py \
  --output-dir ./qwen-websec-finetuned \
  --model-id Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --jsonl-dataset training/websec_repair_train_100.jsonl \
  --trajectory-logs trajectory_logs.json \
  --research-doc training/research_websec_review_playbook.md \
  --external-websec-jsonl training/external_websec_corpus.jsonl \
  --no-include-swe-bench \
  --epochs 3 \
  --learning-rate 2e-4 \
  --batch-size 4 \
  --gradient-accumulation-steps 4
```

This will:
1. Load Qwen2.5-Coder-1.5B-Instruct (base model)
2. Apply 4-bit quantization (saves memory)
3. Attach LoRA adapters (r=16, 32 layers)
4. Build a mixed dataset from your WebSec JSONL, environment trajectories, synthetic tasks, the research playbook, and the optional external WebSec corpus
5. Save the fine-tuned adapter

Expected time: 15-30 minutes

Expected output:
```
========================================
WebSec Repair - QLoRA Training
========================================
Model: Qwen/Qwen2.5-Coder-1.5B-Instruct
Output: ./qwen-websec-finetuned
GPU: NVIDIA A100-SXM4-40GB
GPU Memory: 40.0 GB

1. Loading training data...
   Total examples: 594
   Source breakdown:
     - trajectory: 250
     - research_doc: 238
     - websec_jsonl: 100
     - primevul: 24
     - devign: 24
     - juliet: 18
     - owasp_benchmark: 18
     - bigvul: 12
     - github_security_fix: 12
     - synthetic: 6
   Train split: 535
   Eval split: 59

2. Loading model and tokenizer...
   Model loaded (params: 1,541,456,896)

3. Configuring LoRA...
   Trainable params: 2,752,512
   Total params: 1,541,456,896
   Trainable ratio: 0.18%

... training progress ...

Training complete!
========================================
```

#### Cell 3: Test the model

```python
from training.inference import WebSecRepairAgent
import json

# Load fine-tuned model
agent = WebSecRepairAgent("./qwen-websec-finetuned")

# Test on SQL injection
action = agent.get_action(
    task_id="sqli_login",
    instruction="Fix SQL injection",
    code_snippet='query = f"SELECT * FROM users WHERE id={uid}"',
    scanner_hint="SQL injection detected"
)

print("Model recommendation:")
print(json.dumps(action, indent=2))
```

Expected output (environment-compatible action is nested under `action`):
```json
{
  "action": {
    "action_type": "classify",
    "vulnerability_type": "sql_injection"
  },
  "review": {
    "owasp_top10_2021": "A03: Injection",
    "cwe_id": "CWE-89",
    "rationale": "User input changes SQL query semantics."
  },
  "action_type": "classify",
  "vulnerability_type": "sql_injection"
}
```

If you need only the current environment action block, use `agent.get_env_action(...)`.

### Dataset Priorities

The Colab training flow now expects this order of importance:

1. `training/websec_repair_train_100.jsonl`
2. `trajectory_logs.json`
3. `training/external_websec_corpus.jsonl`
4. `training/research_websec_review_playbook.md`
5. SWE-bench only if you explicitly pass `--include-swe-bench`

The trainer refuses to run on fewer than 50 total examples unless you pass
`--allow-tiny-debug-dataset`.

### Save to Hugging Face Hub (Optional)

To save your trained model to Hugging Face:

```bash
# In Colab, after training
!huggingface-cli login  # Paste your HF token

# Or run training with HF hub integration
!python training/qlora_train.py \
  --output-dir ./qwen-websec-finetuned \
  --hf-model-id YOUR_USERNAME/qwen-websec \
  --epochs 3
```

This pushes your model to `huggingface.co/YOUR_USERNAME/qwen-websec`

---

## Model Variants

### Use Qwen2.5-Coder-3B instead

For a more powerful model (slower, but better accuracy):

```bash
!python training/qlora_train.py \
  --output-dir ./qwen-websec-3b \
  --model-id Qwen/Qwen2.5-Coder-3B-Instruct \
  --epochs 2 \
  --batch-size 2
```

Note: 3B model needs smaller batch size due to memory.

### Use Qwen2.5-Coder-7B (local or cloud GPU only)

```bash
# Only for local A100/H100 or paid cloud
!python training/qlora_train.py \
  --model-id Qwen/Qwen2.5-Coder-7B-Instruct \
  --batch-size 1 \
  --epochs 1
```

---

## Troubleshooting

### "CUDA out of memory"
- Reduce `--batch-size` (try 2 or 1)
- Reduce `--gradient-accumulation-steps` (try 2)
- Use smaller model (`--model-id Qwen/Qwen2.5-Coder-1.5B-Instruct`)

### "No GPU available"
- Check Colab runtime type: Runtime > Change runtime type > GPU
- Verify with `!nvidia-smi`

### "ModuleNotFoundError: No module named 'training'"
- Make sure you're in the correct directory: `%cd websec_repair_env`
- Or adjust PYTHONPATH: `import sys; sys.path.insert(0, '.')`

### "Model loading is slow"
- First time loading downloads ~3GB
- Subsequent runs use cached weights
- Consider saving to Hugging Face for faster loading

---

## Next: Connect to RL Loop

Once trained, integrate with your environment:

```python
from websec_repair_env import WebSecRepairEnv
from training.inference import WebSecRepairAgent

# Initialize
env = WebSecRepairEnv()
agent = WebSecRepairAgent("./qwen-websec-finetuned")

# Run episode with trained agent
for episode in range(10):
    obs = env.reset()
    done = False
    
    while not done:
        # Get action from fine-tuned model
        action = agent.get_action(
            task_id=obs["task_id"],
            instruction=obs["instruction"],
            code_snippet=obs["code_snippet"],
            scanner_hint=obs["scanner_hint"],
        )
        
        # Execute in environment
        obs, reward, done, info = env.step(action)
        
        print(f"Reward: {reward}, Done: {done}")
```

---

## Advanced: Custom Training Data

To train on your own vulnerability data:

1. **Export your environment trajectories**:
   ```python
   # From your environment run logs
   import json
   with open("trajectory_logs.json") as f:
       trajectories = json.load(f)
   ```

2. **Update `data_loader.py`**:
   ```python
   def load_trajectory_data(trajectory_file):
       # Parse your logs into (instruction, response) pairs
       ...
   ```

3. **Retrain with custom data**:
   ```bash
   python training/qlora_train.py --data-file trajectory_logs.json
   ```

---

## Production Deployment

### Save Model Locally
```bash
# After training, download from Colab
from google.colab import files
files.download("qwen-websec-finetuned/adapter/adapter_model.safetensors")
```

### Load in Production
```python
from training.inference import WebSecRepairAgent

agent = WebSecRepairAgent("/path/to/qwen-websec-finetuned")
action = agent.get_action(task_id, instruction, code, hint)
```

### Quantize for Inference
```python
# Use the same 4-bit quantization for faster inference
from transformers import BitsAndBytesConfig
```

---

## Cost Estimate

- **Colab Free**: $0 (but limited to 12 hours/session)
- **Colab Pro**: $10/month (24+ hour sessions)
- **RunPod** (GPU rental): ~$0.30/hour for 3B model training
- **Lambda Labs** (GPU rental): ~$0.50/hour for 3B model training

For full training on Colab Free: ~2-3 sessions of 12 hours

---

## Resources

- [Qwen2.5-Coder Models](https://huggingface.co/Qwen)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [PEFT Library](https://github.com/huggingface/peft)
- [Hugging Face Fine-tuning Guide](https://huggingface.co/docs/transformers/training)

---

## Questions?

If training fails:
1. Check GPU availability: `!nvidia-smi`
2. Check memory usage during training
3. Try reducing batch size
4. Check Hugging Face model downloads aren't blocked
5. See error logs in Colab output

Good luck! 🚀
