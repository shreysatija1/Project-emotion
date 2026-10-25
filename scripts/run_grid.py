"""
Experiment Grid Runner for Speech Emotion Recognition 1D-CNN.

Grid:
  - 2 Feature types: MFCC, MFMC
  - 3 Coefficient sizes: 12, 24, 30
  - 3 Augmentation modes: none, after_split, before_split
  Total: 18 configurations per seed.
  Seeds: 42 (primary), followed by 43 and 44.

Features:
  - Fully resumable (skips already completed runs with valid summary & model checkpoint).
  - Updates PROGRESS.md after each run with run name, val accuracy, and elapsed time.
  - Runs in-process or via CLI.
"""

import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import pandas as pd
from config import RESULTS_DIR, ROOT
from train import train_run


def init_progress_file(progress_path):
    """Initialize PROGRESS.md table if it does not exist."""
    progress_path = Path(progress_path)
    if not progress_path.exists():
        header = (
            "# Week 3 Training Progress Log\n\n"
            "| # | Run Name | Feature | Coeffs | Aug Mode | Seed | Epochs | Val Acc | Val Loss | Time (s) | Status |\n"
            "| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        )
        progress_path.write_text(header, encoding="utf-8")


def append_progress_entry(progress_path, idx, summary, status="Completed"):
    """Append a completed or skipped run to PROGRESS.md."""
    progress_path = Path(progress_path)
    run_name = summary["run_name"]
    feat = summary["feature_type"].upper()
    n = summary["n_coeff"]
    mode = summary["aug_mode"]
    seed = summary["seed"]
    epochs = summary.get("epochs_trained", 100)
    val_acc = f"{summary.get('val_accuracy', 0.0) * 100:.2f}%"
    val_loss = f"{summary.get('val_loss', 0.0):.4f}"
    elapsed = f"{summary.get('elapsed_seconds', 0.0):.1f}"

    line = f"| {idx} | `{run_name}` | {feat} | {n} | `{mode}` | {seed} | {epochs} | **{val_acc}** | {val_loss} | {elapsed}s | {status} |\n"
    with open(progress_path, "a", encoding="utf-8") as f:
        f.write(line)


def main():
    results_dir = RESULTS_DIR
    models_dir = ROOT / "models"
    progress_path = ROOT / "PROGRESS.md"

    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    init_progress_file(progress_path)

    feature_types = ["mfcc", "mfmc"]
    coeff_sizes = [30, 24, 12]  # priority order
    aug_modes = ["after_split", "none", "before_split"]
    seeds = [42, 43, 44]

    # Generate all grid combinations
    # Priority: Seed 42 for all 18 first, then seeds 43 and 44
    runs = []
    for s in seeds:
        for c in coeff_sizes:
            for m in aug_modes:
                for f in feature_types:
                    runs.append((f, c, m, s))

    print(f"Total grid runs planned: {len(runs)} (18 configs x {len(seeds)} seeds)")
    total_start_time = time.time()
    completed_count = 0

    for idx, (f_type, n_coeff, mode, seed) in enumerate(runs, start=1):
        run_name = f"{f_type}_{n_coeff}_{mode}_seed{seed}"
        summary_path = results_dir / f"summary_{run_name}.json"
        model_path = models_dir / f"best_model_{run_name}.keras"

        # Resumability check
        if summary_path.exists() and model_path.exists():
            print(f"[{idx}/{len(runs)}] SKIPPING existing run: {run_name}")
            import json
            with open(summary_path, "r") as f:
                saved_summary = json.load(f)
            append_progress_entry(progress_path, idx, saved_summary, status="Cached")
            continue

        print(f"\n=======================================================")
        print(f"[{idx}/{len(runs)}] Executing Grid Run: {run_name}")
        print(f"=======================================================")

        summary = train_run(
            feature_type=f_type,
            n_coeff=n_coeff,
            aug_mode=mode,
            seed=seed,
            epochs=100,
            batch_size=32,
            lr=0.001,
            early_stopping=False,
            results_dir=results_dir,
            models_dir=models_dir,
        )

        completed_count += 1
        append_progress_entry(progress_path, idx, summary, status="Completed")

    total_elapsed = round(time.time() - total_start_time, 2)
    print(f"\n=======================================================")
    print(f"EXPERIMENT GRID FINISHED!")
    print(f"Completed {completed_count} new runs. Total time: {total_elapsed}s ({total_elapsed/60:.2f} min)")
    print(f"Progress log available at: {progress_path}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
