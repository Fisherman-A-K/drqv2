# CSC415 A1 — DrQ-v2 Reproduction

Reproduction of **"Mastering Visual Continuous Control: Improved Data-Augmented
Reinforcement Learning"** (Yarats et al., 2022) for CSC415 Assignment 1.

Original repository: <https://github.com/facebookresearch/drqv2>

---

## Project Description

DrQ-v2 is an off-policy, model-free RL algorithm for **pixel-based continuous
control**.  It builds on DrQ by:

- Replacing SAC with DDPG as the base RL algorithm.
- Using **n-step returns** (default n = 3) for TD target estimation.
- A **decaying exploration-noise schedule** (σ: 1.0 → 0.1 over 500 k frames).
- Random-shift data augmentation applied on the GPU.

This repo reproduces the main paper results on `walker_walk` and `cheetah_run`,
and runs an **n-step return ablation** study (n ∈ {1, 3, 5}) on `walker_walk`.

---

## Setup

### 1. Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10 |
| OS | Windows 10/11, Ubuntu 20.04+, macOS 12+ |
| GPU (optional) | CUDA-capable NVIDIA GPU (CPU fallback available) |

### 2. Create and activate the virtual environment

```bash
# Create venv (run once)
python -m venv venv

# Activate — Linux/macOS
source venv/bin/activate

# Activate — Windows (Git Bash / MSYS2)
source venv/Scripts/activate

# Activate — Windows (Command Prompt / PowerShell)
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on MuJoCo:** the new `mujoco` pip package (2.3.7) is used directly —
> no manual MuJoCo installation or license key is required.

### 4. Verify the setup

```bash
python -c "
import dmc, drqv2, torch
env = dmc.make('walker_walk', frame_stack=3, action_repeat=2, seed=1)
print('obs shape:', env.observation_spec().shape)
print('PyTorch:', torch.__version__)
print('Setup OK!')
"
```

---

## How to Run Training

### Single run (manual)

```bash
python train.py \
    "task@_global_=walker_walk" \
    seed=1 \
    device=cpu \
    num_train_frames=500000 \
    save_video=false \
    use_tb=false \
    replay_buffer_num_workers=0 \
    hydra.run.dir=results/main/walker_walk/seed_1
```

Replace `device=cpu` with `device=cuda` if a GPU is available.

### All main experiments (3 seeds x 2 tasks)

```bash
bash scripts/run_main_experiments.sh
```

### Ablation study (n-step return, 3 seeds x 3 variants)

```bash
bash scripts/run_ablation.sh
```

---

## How to Generate Plots

After training is complete:

```bash
python scripts/generate_plots.py
```

Plots are saved to `results/plots/` as both `.png` and `.pdf`.

| File | Description |
|---|---|
| `main_results.png` | Learning curves for all main tasks |
| `ablation_nstep.png` | N-step return ablation on walker_walk |

---

## File Structure

```
drqv2/
├── README.md                         <- This file
├── requirements.txt                  <- Pinned dependencies
├── train.py                          <- Main training script (Hydra entry point)
├── drqv2.py                          <- DrQ-v2 agent (encoder, actor, critic)
├── dmc.py                            <- DeepMind Control Suite wrappers
├── replay_buffer.py                  <- Replay buffer with n-step returns
├── logger.py                         <- CSV + TensorBoard logging
├── utils.py                          <- Utilities (schedules, soft updates, ...)
├── video.py                          <- Video recording helpers
├── cfgs/
│   ├── config.yaml                   <- Main Hydra config
│   └── task/                         <- Per-task overrides (27 tasks)
│       ├── easy.yaml                 <- 1.1 M frames, sigma schedule
│       ├── medium.yaml               <- 3.1 M frames, sigma schedule
│       ├── walker_walk.yaml
│       ├── cheetah_run.yaml
│       └── ...
├── scripts/
│   ├── run_main_experiments.sh       <- Launch all main experiments
│   ├── run_ablation.sh               <- Launch ablation study
│   └── generate_plots.py             <- Generate all figures
├── results/
│   ├── main/
│   │   ├── walker_walk/seed_1/       <- eval.csv, train.csv, buffer/, ...
│   │   └── ...
│   ├── ablation/
│   │   ├── nstep_1/seed_1/
│   │   ├── nstep_3/seed_1/
│   │   └── nstep_5/seed_1/
│   └── plots/
│       ├── main_results.png / .pdf
│       └── ablation_nstep.png / .pdf
└── curves/                           <- Paper benchmark curves (original repo)
```

---

## Hyperparameters

The following table lists the hyperparameters used for all main experiments.
Values match the paper defaults unless otherwise noted.

| Hyperparameter | Value | Notes |
|---|---|---|
| Frame stack | 3 | 3 consecutive 84x84 RGB frames |
| Action repeat | 2 | |
| Discount gamma | 0.99 | |
| N-step returns | 1 (walker_walk) / 3 (cheetah_run) | Per task config |
| Batch size | 512 (walker_walk) / 256 (cheetah_run) | Per task config |
| Replay buffer size | 1 000 000 | |
| Seed frames | 4 000 | Random exploration before learning |
| Exploration steps | 2 000 | Uniform random actions at start |
| Stddev schedule | linear(1.0, 0.1, 100k) easy / linear(1.0, 0.1, 500k) medium | Decaying noise |
| Stddev clip | 0.3 | |
| Learning rate | 1e-4 | Actor, critic, encoder |
| Hidden dim | 1 024 | MLP layers in actor/critic |
| Feature dim | 50 | Encoder output dimension |
| Critic target tau | 0.01 | Soft update rate |
| Update frequency | Every 2 env steps | |
| Evaluation frequency | Every 10 000 frames | |
| Eval episodes | 10 | |
| Training frames | 500 000* | Reduced from paper (1.1 M / 3.1 M) for compute |
| Device | cpu (auto cuda if available) | |

*Training frames were reduced from the paper defaults for practical compute
 constraints.  The learning-curve shape accurately reflects the algorithm
 behaviour; absolute final performance may be lower than paper numbers.

---

## Ablation Study

**Hypothesis:** Using n = 3 step returns (the DrQ-v2 default) yields better
sample efficiency than n = 1 (standard TD) because multi-step returns reduce
variance and propagate reward signals faster through the replay buffer.
Larger n (n = 5) may slightly hurt performance due to increased bias from
off-policy bootstrapping, but the effect depends on the task horizon.

**Task:** `walker_walk` with 3 random seeds per variant.

**Variants compared:**

| Variant | N-step | Notes |
|---|---|---|
| `nstep_1` | 1 | Standard 1-step TD (Bellman) |
| `nstep_3` | 3 | DrQ-v2 default |
| `nstep_5` | 5 | Longer return horizon |

---

## Modifications to Original Code

The following changes were made to the original DrQ-v2 code for
Python 3.10 / PyTorch 2.x / Windows compatibility.  Core algorithm
logic is **unchanged**.

| File | Change | Reason |
|---|---|---|
| `train.py` | Set `MUJOCO_GL=glfw` on Windows/macOS, skip `MKL_SERVICE_FORCE_INTEL` on non-Linux | EGL not available on Windows |
| `train.py` | Added `version_base='1.1'` to `@hydra.main` | Suppresses hydra 1.3 deprecation warning |
| `drqv2.py` | Changed `.cpu().numpy()` to `.cpu().detach().numpy()` in `act()` | PyTorch 2.x requires `.detach()` before `.numpy()` |
| `cfgs/config.yaml` | Removed `- override hydra/launcher: submitit_local` | submitit is for SLURM clusters; not needed locally |
| `cfgs/config.yaml` | Changed `device: cuda` to `device: cpu` (default) | Safe fallback for machines without GPU |
| `cfgs/config.yaml` | Changed `replay_buffer_num_workers: 4` to `0` | Avoids Windows DataLoader multiprocessing issues |
| `cfgs/config.yaml` | Removed SLURM launcher config block | Not applicable for local runs |

---

## Original Paper

```bibtex
@article{yarats2021drqv2,
  title={Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning},
  author={Denis Yarats and Rob Fergus and Alessandro Lazaric and Lerrel Pinto},
  journal={arXiv preprint arXiv:2107.09645},
  year={2021}
}
```

## License

MIT License — see [LICENSE](LICENSE).
Original implementation Copyright (c) Facebook, Inc. and its affiliates.
