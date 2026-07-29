# Franka Multi-Fruit Sorting via SmolVLA on AMD ROCm

> **Track 3 — Physical AI Challenge** | AMD Radeon Hackathon 2026
> Team: ROCm Robotics | Application: Franka Multi-Fruit Sorting

This project demonstrates a Franka panda robotic arm learning to sort multiple fruits (plum, banana, lemon) into color-matched bowls in a Genesis physics simulator. The SmolVLA Vision-Language-Action model is fine-tuned on AMD Radeon ROCm 7.2.1 and achieves 100% closed-loop evaluation success rate.

## Table of Contents

- [1. Environment Setup](#1-environment-setup)
- [2. Execution and Usage](#2-execution-and-usage)
- [3. Dependency Specifications](#3-dependency-specifications)
- [4. Step-by-Step Reproduction](#4-step-by-step-reproduction)
- [Project Structure](#project-structure)
- [Results](#results)

---

## 1. Environment Setup

### 1.1 Prerequisites

- AMD Radeon GPU (tested on Radeon Pro W7900 48GB via Radeon Cloud)
- AMD ROCm 7.2.1 or later
- Ubuntu 22.04 LTS (or Radeon Cloud GPU instance)
- Python 3.10
- Git

### 1.2 Provision a Radeon Cloud Instance

1. Log in to [Radeon Cloud](https://radeon-cloud.amd.com/) using your registered email.
2. Create a new GPU instance with the following spec:
   - GPU: Radeon Pro W7900 (48GB)
   - Image: ROCm 7.2.1 + PyTorch 2.9.1 base
   - Disk: ≥ 100 GB
3. Wait for the instance to become `Running`, then SSH or open JupyterLab.

### 1.3 Verify ROCm Installation

```bash
rocm-smi
# Expected: GPU status, 48GB memory, ROCm 7.2.1

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.hip)"
# Expected: 2.9.1+rocm7.2.1 True 7.2.1
```

### 1.4 Set Environment Variables

```bash
export MPLBACKEND=Agg          # Headless rendering (no display)
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/workspace/.cache/huggingface
```

Add these to `~/.bashrc` for persistence.

### 1.5 Clone Repositories

```bash
cd /workspace

# Official starter repository (provides Genesis scene + Franka control modules)
git clone https://github.com/wangxunx/franka_fruit_pick_demo.git

# This project's submission
# (Replace <your-fork> with your GitHub fork URL)
git clone https://github.com/<your-fork>/Radeon-hackathon-2026-07.git amd-submission
```

---

## 2. Execution and Usage

### 2.1 Run the Scripted Demo (No Training)

This builds the sort-task scene (3 fruits + 3 color bowls) and runs a scripted pick-and-place sequence. Useful for verifying the Genesis + Franka setup works.

```bash
cd /workspace/amd-submission
PYTHONPATH=/workspace:/workspace/franka_fruit_pick_demo:$PYTHONPATH \
    python submission/src/scene/sort_demo.py --save-frames
# Outputs: sort_final.png, after_*_*.png in working directory
```

### 2.2 Collect a Demonstration Dataset

Run the scripted policy to collect LeRobot-format episodes:

```bash
python submission/src/data/record_sort_dataset.py \
    --episodes 5 \
    --fps 30 \
    --img-wh 224 224
# Outputs: /workspace/data/<repo_id>/ dataset in LeRobot format
```

### 2.3 Fine-Tune SmolVLA

```bash
python submission/src/train/run_smolvla_train.py
# Background process; logs to /tmp/smolvla_train.log
# Monitor with:
python submission/src/train/monitor_train.py
```

Key training arguments (set inside `run_smolvla_train.py`):
- `--dataset.repo_id local/sort_fruit`
- `--policy.pretrained_path=lerobot/smolvla_base`
- `--rename_map '{"camera1":"world","camera2":"wrist"}'`
- `--batch_size 4`
- `--steps 2000`

### 2.4 Run Closed-Loop Evaluation

```bash
python submission/src/eval/eval_sort_smolvla.py \
    --checkpoint 002000 \
    --episodes 1
# Outputs: ep1_<id>_<fruit>.mp4 videos, eval_results.json
```

---

## 3. Dependency Specifications

### 3.1 Python Dependencies

See [submission/requirements.txt](./submission/requirements.txt) for the pinned list.

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.9.1+rocm7.2.1 | Deep learning (ROCm build) |
| genesis-world | 1.2.3 | Physics simulation |
| lerobot | 0.6.0 | Robot learning framework |
| transformers | ≥4.40.0 | SmolVLA backbone |
| opencv-python-headless | ≥4.8.0 | Image processing |
| playwright | ≥1.40.0 | Remote execution utilities |

### 3.2 System Dependencies

- ROCm 7.2.1 (provides `rocm-smi`, HIP runtime)
- libGL1, libglib2.0-0 (for OpenCV headless)
- curl, wget, git

### 3.3 Hardware

| Resource | Minimum | Tested |
|----------|---------|--------|
| GPU | Radeon Pro W7900 48GB | Radeon Pro W7900 48GB |
| VRAM | 32 GB | 48 GB |
| Disk | 50 GB | 100 GB |
| RAM | 16 GB | 32 GB |

---

## 4. Step-by-Step Reproduction

Reproduces the 100% evaluation success rate reported in `logs/eval_results.json`.

### Step 1 — Provision Instance & Clone Repos

Follow [Section 1.2–1.5](#12-provision-a-radeon-cloud-instance).

### Step 2 — Install Python Dependencies

```bash
# PyTorch with ROCm
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/rocm7.2.1

# Project dependencies
pip install -r submission/requirements.txt

# Playwright browser for remote utils (optional)
python -m playwright install chromium --with-deps
```

### Step 3 — Verify Genesis + Franka Setup

```bash
PYTHONPATH=/workspace:/workspace/franka_fruit_pick_demo:$PYTHONPATH \
    python submission/src/scene/sort_demo.py --save-frames
```

Expected: A `sort_final.png` appears showing 3 fruits placed in 3 colored bowls.

### Step 4 — Collect Training Dataset

```bash
python submission/src/data/record_sort_dataset.py \
    --episodes 5 \
    --fps 30 \
    --img-wh 224 224
```

Expected: A LeRobot dataset at `/workspace/data/local/sort_fruit/` containing 5 successful episodes with `observation.images.world`, `observation.images.wrist`, `observation.state`, and `action` keys.

### Step 5 — Fine-Tune SmolVLA

```bash
python submission/src/train/run_smolvla_train.py
python submission/src/train/monitor_train.py
```

Expected training trajectory (reference data from `logs/smolvla_train.log`):
- 2000 steps total
- Final training loss: 0.073
- Wall-clock time: ~5 minutes on W7900
- Checkpoint saved at `outputs/train/smolvla_sort_fruit/checkpoints/002000/`

### Step 6 — Run Closed-Loop Evaluation

```bash
python submission/src/eval/eval_sort_smolvla.py \
    --checkpoint 002000 \
    --episodes 1
```

Expected output (matches `logs/eval_results.json`):

```json
{
  "checkpoint": "002000",
  "total_episodes": 2,
  "successful_episodes": 2,
  "success_rate": 1.0,
  "episodes": [
    {"id": "018_plum", "fruit": "plum", "target_bowl": "purple", "success": true, "steps": 308},
    {"id": "011_banana", "fruit": "banana", "target_bowl": "yellow", "success": true, "steps": 221}
  ]
}
```

Expected videos: `ep1_018_plum.mp4`, `ep1_011_banana.mp4` (also stored under `screenshots/sort_demo/`).

### Step 7 — Verify Results

Compare your `eval_results.json` against the reference in `logs/eval_results.json`. Success rate should be 100% (2/2 episodes).

---

## Project Structure

```
.
├── README.md                  # This file (reproducibility guide)
├── Dockerfile                 # Container image definition (preferable)
├── .dockerignore
├── submission/                # Source code (see submission/README.md)
│   ├── src/{scene,data,train,eval,utils}/
│   ├── configs/sort_scene_config.py
│   └── requirements.txt
├── docs/                      # Submission artifacts
│   ├── Technical_Report.pdf   # 8-section technical report
│   ├── demo_video.mp4         # 3-5 minute demo video
│   └── upstream_contribution_evidence.png  # Open-source contribution proof
├── logs/                      # Reference training logs & eval results
│   ├── smolvla_train.log
│   └── eval_results.json
└── screenshots/sort_demo/     # Demo video source materials
```

## Results

| Metric | Value |
|--------|-------|
| Closed-loop success rate | **100% (2/2 episodes)** |
| Training steps | 2000 |
| Final training loss | 0.073 |
| Training wall-clock | ~5 minutes (W7900) |
| Fruits sorted | plum ✓, banana ✓ (lemon acts as distractor) |
| Checkpoint | `002000` |

## License

This project uses open-source components:
- Genesis (Apache-2.0)
- LeRobot (Apache-2.0)
- SmolVLA (Apache-2.0)

The submission code itself is licensed under MIT.

## Upstream Contribution

As part of this hackathon, we contributed to the open-source community by filing an issue on `huggingface/lerobot` reporting a ROCm-specific bug in uint8 tensor bilinear interpolation:
- Issue #4205: [ROCm] NotImplementedError for bilinear interpolate on uint8 tensor in SmolVLA resize_with_pad during manual inference
- URL: https://github.com/huggingface/lerobot/issues/4205
