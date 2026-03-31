# WebSec Repair Environment - How to Run Everything

## ⚠️ Important: Correct Server Startup Command

The server needs to be run with the full module path:

```bash
# CORRECT ✅
uv run python -m websec_repair_env.server.app

# WRONG ❌ (will fail with "No module named 'server'")
uv run uvicorn server.app:app
```

## 🚀 Full Workflow

### Step 1: Start the Server
```bash
cd websec_repair_env

# Start in background (or use a separate terminal)
uv run python -m websec_repair_env.server.app &

# Server will start on http://127.0.0.1:8000
# Wait for: "Uvicorn running on http://0.0.0.0:8000"
```

### Step 2: Test the Web UI
Open in browser: http://127.0.0.1:8000

You should see the dark blue WebSec Repair dashboard.

### Step 3: Run Baseline Agent (15 episodes with metrics)
```bash
uv run python inference_with_metrics.py \
  --base-url http://127.0.0.1:8000 \
  --task sqli_login \
  --episodes 15 \
  --metrics \
  --output-dir baseline_run
```

Expected output:
```
Success Rate: 100.0%
Vulnerability Accuracy: 0.0%
Patch Accuracy: 100.0%
✅ Metrics saved to baseline_run/metrics.json
```

### Step 4: Run Trained Agent (15 episodes with metrics)
```bash
uv run python trained_agent.py \
  --base-url http://127.0.0.1:8000 \
  --task sqli_login \
  --episodes 15 \
  --metrics \
  --output-dir trained_run
```

Expected output:
```
Success Rate: 100.0%
Vulnerability Accuracy: 100.0%  ← Improved!
Patch Accuracy: 100.0%
✅ Metrics saved to trained_run/metrics.json
```

### Step 5: Generate Comparison Report
```bash
uv run python generate_comparison.py
```

Output:
```
📊 GENERATING COMPARISON REPORT
======================================================================

📈 BASELINE AGENT
  Success Rate:          100.0%
  Vulnerability Acc:       0.0%
  Patch Selection Acc:   100.0%

📈 TRAINED AGENT
  Success Rate:          100.0%
  Vulnerability Acc:     100.0%
  Patch Selection Acc:   100.0%

🎯 IMPROVEMENTS
  Vulnerability Acc:    +100.0%

👉 Open report.html in your browser!
```

### Step 6: View Results
```bash
open training_comparison_results/report.html
```

This opens the interactive comparison report with:
- Key findings summary
- Metrics cards (base vs trained)
- 4 matplotlib graphs
  - Success rate comparison
  - Reward trajectory
  - Average metrics
  - Accuracy metrics (shows +100% improvement)

## 📊 What You'll See

The HTML report shows:
- **Base Model**: 100% success, **0% vulnerability accuracy**
- **Trained Model**: 100% success, **100% vulnerability accuracy**
- **Improvement**: **+100% in vulnerability classification**

4 comparison plots clearly show the improvement visually.

## 🔧 Troubleshooting

**Error: "ModuleNotFoundError: No module named 'server'"**
- ❌ Wrong: `uv run uvicorn server.app:app`
- ✅ Right: `uv run python -m websec_repair_env.server.app`

**Server won't start on port 8000**
- Check if another server is running: `lsof -i :8000`
- Use a different port: `uv run python -m websec_repair_env.server.app --port 8001`

**Can't connect to server from agents**
- Make sure server is running: `curl http://127.0.0.1:8000/`
- Check firewall/network settings
- Use same host/port in all commands

**Visualization plots not generated**
- Make sure matplotlib is installed: `uv pip install matplotlib`
- Check output directory permissions: `ls -la training_comparison_results/`

## 📁 Output Files

After running all steps:
```
websec_repair_env/
├── baseline_run/
│   └── metrics.json              (15 baseline episodes)
│
├── trained_run/
│   └── metrics.json              (15 trained episodes)
│
└── training_comparison_results/
    ├── report.html               (Interactive report) ← Open this!
    ├── 01_success_rate.png
    ├── 02_reward_trajectory.png
    ├── 03_average_metrics.png
    └── 04_accuracy_comparison.png
```

## 🎯 Quick Start (One Command Each)

```bash
# Terminal 1: Start server
uv run python -m websec_repair_env.server.app

# Terminal 2: Run baseline
uv run python inference_with_metrics.py --episodes 15 --metrics --output-dir baseline_run

# Terminal 3: Run trained agent
uv run python trained_agent.py --episodes 15 --metrics --output-dir trained_run

# Terminal 4: Generate report and view
uv run python generate_comparison.py && open training_comparison_results/report.html
```

All complete! 🎊
