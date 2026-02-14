"""
generate_plots.py
=================
CSC415 A1 - DrQ-v2 Reproduction: Plot Generation

Reads training logs produced by DrQ-v2 (eval.csv files), aggregates
results across seeds, and produces publication-quality figures:

  Plot 1 — Main Results:  learning curves for each task (mean ± std).
  Plot 2 — Ablation:      learning curves for each n-step variant (mean ± std).

Usage (from repo root):
    python scripts/generate_plots.py

Outputs are saved to:
    results/plots/main_results.png / .pdf
    results/plots/ablation_nstep.png / .pdf
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Plotting style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'lines.linewidth': 2.0,
})

# Colour palette — one colour per condition
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_eval_csv(seed_dir: Path) -> pd.DataFrame | None:
    """Load eval.csv from a single seed directory.  Returns None if missing."""
    csv_path = seed_dir / 'eval.csv'
    if not csv_path.exists():
        print(f'  [warn] No eval.csv in {seed_dir}')
        return None
    df = pd.read_csv(csv_path)
    # Normalise column names (older logs may use 'frame' instead of 'step')
    df.columns = [c.strip() for c in df.columns]
    return df


def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply a uniform rolling average for readability."""
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    # 'same' mode keeps the same length; 'valid' would shorten the array
    return np.convolve(values, kernel, mode='same')


def load_seeds(base_dir: Path, seed_prefix: str = 'seed_') -> list[pd.DataFrame]:
    """Collect DataFrames from all seed sub-directories under base_dir."""
    dfs = []
    seed_dirs = sorted(base_dir.glob(f'{seed_prefix}*'))
    if not seed_dirs:
        # Also support a flat layout where each run is the base dir itself
        df = load_eval_csv(base_dir)
        if df is not None:
            dfs.append(df)
        return dfs
    for sd in seed_dirs:
        df = load_eval_csv(sd)
        if df is not None:
            dfs.append(df)
    return dfs


def align_and_aggregate(dfs: list[pd.DataFrame],
                         x_col: str = 'frame',
                         y_col: str = 'episode_reward'
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate each seed's curve onto a common x-axis, then compute
    mean and standard deviation across seeds.

    Returns (x_common, mean_y, std_y).
    """
    # Find the common x column (drqv2 logs use 'frame' in eval.csv)
    for candidate in ('frame', 'step', 'Step', 'Frame', 'env_step'):
        if candidate in dfs[0].columns:
            x_col = candidate
            break

    # Find the common y column
    for candidate in ('episode_reward', 'eval_episode_reward', 'reward',
                       'episode_return', 'return'):
        if candidate in dfs[0].columns:
            y_col = candidate
            break

    # Common x range = intersection of all seed x ranges
    x_min = max(df[x_col].min() for df in dfs)
    x_max = min(df[x_col].max() for df in dfs)
    x_common = np.linspace(x_min, x_max, num=200)

    interpolated = []
    for df in dfs:
        xs = df[x_col].values.astype(float)
        ys = df[y_col].values.astype(float)
        # Drop duplicate x values by taking the mean
        unique_x, idx = np.unique(xs, return_index=True)
        unique_y = ys[idx]
        y_interp = np.interp(x_common, unique_x, unique_y)
        interpolated.append(y_interp)

    arr = np.stack(interpolated, axis=0)  # shape: (num_seeds, num_points)
    return x_common, arr.mean(axis=0), arr.std(axis=0)


# ---------------------------------------------------------------------------
# Paper reference values (DrQ-v2 paper Table 1, 1M steps)
# Used as horizontal reference lines.
# ---------------------------------------------------------------------------
PAPER_REFERENCE = {
    'walker_walk': 948,
    'cheetah_run': 660,
    'cartpole_swingup': 869,
}


# ---------------------------------------------------------------------------
# Plot 1: Main Results
# ---------------------------------------------------------------------------

def plot_main_results(results_dir: Path, plots_dir: Path,
                      smooth_window: int = 5) -> None:
    """
    Plot learning curves for all tasks in results/main/.
    One subplot per task; mean ± std across seeds.
    """
    main_dir = results_dir / 'main'
    if not main_dir.exists():
        print(f'[info] {main_dir} does not exist — skipping main results plot.')
        return

    # Collect available tasks
    tasks = sorted([d.name for d in main_dir.iterdir() if d.is_dir()])
    if not tasks:
        print(f'[info] No task directories found in {main_dir}.')
        return

    n_tasks = len(tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 4), squeeze=False)
    fig.suptitle('DrQ-v2 Reproduction — Main Results', fontsize=14, y=1.02)

    for ax, task in zip(axes[0], tasks):
        task_dir = main_dir / task
        dfs = load_seeds(task_dir)

        if not dfs:
            ax.set_title(f'{task}\n(no data)')
            ax.set_xlabel('Environment Steps')
            ax.set_ylabel('Episode Return')
            continue

        x, mean_y, std_y = align_and_aggregate(dfs)

        # Smooth for readability
        mean_s = smooth(mean_y, smooth_window)
        std_s = smooth(std_y, smooth_window)

        ax.plot(x, mean_s, color=COLORS[0], label=f'DrQ-v2 (n={len(dfs)} seeds)')
        ax.fill_between(x, mean_s - std_s, mean_s + std_s,
                        alpha=0.2, color=COLORS[0])

        # Draw paper reference line if available
        ref = PAPER_REFERENCE.get(task)
        if ref is not None:
            ax.axhline(ref, color='grey', linestyle='--', linewidth=1.2,
                       label=f'Paper (1M steps): {ref}')

        ax.set_title(task.replace('_', ' ').title())
        ax.set_xlabel('Environment Steps')
        ax.set_ylabel('Episode Return')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Report final mean ± std
        print(f'  {task}: final return = {mean_y[-1]:.1f} ± {std_y[-1]:.1f} '
              f'(mean over {len(dfs)} seed(s))')

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        out_path = plots_dir / f'main_results.{ext}'
        fig.savefig(out_path, bbox_inches='tight')
        print(f'  Saved: {out_path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Ablation Study (N-step Return)
# ---------------------------------------------------------------------------

def plot_ablation(results_dir: Path, plots_dir: Path,
                  smooth_window: int = 5) -> None:
    """
    Plot ablation learning curves from results/ablation/.
    Expected directory structure:
        results/ablation/nstep_<N>/seed_<K>/eval.csv
    """
    ablation_dir = results_dir / 'ablation'
    if not ablation_dir.exists():
        print(f'[info] {ablation_dir} does not exist — skipping ablation plot.')
        return

    # Collect variant directories (e.g. nstep_1, nstep_3, nstep_5)
    variants = sorted([d.name for d in ablation_dir.iterdir() if d.is_dir()])
    if not variants:
        print(f'[info] No variant directories found in {ablation_dir}.')
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title('Ablation Study — N-step Return (walker_walk)', fontsize=13)
    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Episode Return')

    for color, variant in zip(COLORS, variants):
        var_dir = ablation_dir / variant
        dfs = load_seeds(var_dir)
        if not dfs:
            print(f'  [warn] No data for variant {variant}')
            continue

        x, mean_y, std_y = align_and_aggregate(dfs)
        mean_s = smooth(mean_y, smooth_window)
        std_s = smooth(std_y, smooth_window)

        # Build a readable label (e.g. "nstep=3 (default)")
        label = variant.replace('_', '=')
        if variant == 'nstep_3':
            label += ' (default)'

        ax.plot(x, mean_s, color=color, label=f'{label} (n={len(dfs)} seeds)')
        ax.fill_between(x, mean_s - std_s, mean_s + std_s,
                        alpha=0.15, color=color)

        print(f'  {variant}: final return = {mean_y[-1]:.1f} ± {std_y[-1]:.1f} '
              f'(mean over {len(dfs)} seed(s))')

    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    for ext in ('png', 'pdf'):
        out_path = plots_dir / f'ablation_nstep.{ext}'
        fig.savefig(out_path, bbox_inches='tight')
        print(f'  Saved: {out_path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate DrQ-v2 reproduction plots.')
    parser.add_argument('--results-dir', type=Path,
                        default=Path(__file__).parent.parent / 'results',
                        help='Root results directory (default: ../results)')
    parser.add_argument('--plots-dir', type=Path, default=None,
                        help='Output directory for plots '
                             '(default: <results-dir>/plots)')
    parser.add_argument('--smooth', type=int, default=5,
                        help='Rolling-average window size (default: 5)')
    args = parser.parse_args()

    results_dir: Path = args.results_dir.resolve()
    plots_dir: Path = (args.plots_dir or results_dir / 'plots').resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f'Results directory : {results_dir}')
    print(f'Plots output dir  : {plots_dir}')
    print()

    print('=== Plot 1: Main Results ===')
    plot_main_results(results_dir, plots_dir, smooth_window=args.smooth)
    print()

    print('=== Plot 2: Ablation (N-step Return) ===')
    plot_ablation(results_dir, plots_dir, smooth_window=args.smooth)
    print()

    print('Done.')


if __name__ == '__main__':
    main()
