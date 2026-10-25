"""
Generate summary tables, comparison metrics, and publication-quality plots for Week 4 results.
Outputs:
- results/week4_summary_table.csv
- results/week4_plot_input_formats.png
- results/week4_plot_datasets.png
- results/week4_plot_speaker_independent.png
- results/WEEK4_FINDINGS_NOTE.md
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
CSV_FILE = RESULTS_DIR / "week4_results.csv"

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def load_and_aggregate_results():
    if not CSV_FILE.is_file():
        print(f"Error: {CSV_FILE} does not exist.")
        return None

    df = pd.read_csv(CSV_FILE)
    # Group by key experimental dimensions
    group_cols = ["dataset", "feature", "coeffs", "format", "setup", "split_type"]
    agg_df = df.groupby(group_cols).agg(
        val_mean=("val_acc", "mean"),
        val_std=("val_acc", "std"),
        test_mean=("test_acc", "mean"),
        test_std=("test_acc", "std"),
        train_mean=("train_acc", "mean"),
        runs=("seed", "count")
    ).reset_index()

    # Fill NaN std for single runs
    agg_df["val_std"] = agg_df["val_std"].fillna(0.0)
    agg_df["test_std"] = agg_df["test_std"].fillna(0.0)
    
    out_table = RESULTS_DIR / "week4_summary_table.csv"
    agg_df.to_csv(out_table, index=False)
    print(f"Aggregated summary table saved to {out_table}")
    return agg_df


def plot_input_format_comparison(df):
    """Plot Task 1: Effect of Input Format on EmoDB (baseline vs frames144 vs stats144)."""
    emodb = df[(df["dataset"] == "emodb") & (df["split_type"] == "stratified")].copy()
    if emodb.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    formats = ["baseline", "frames144", "stats144"]
    palette = {"baseline": "#2563eb", "frames144": "#059669", "stats144": "#d97706"}

    for idx, feat in enumerate(["mfcc", "mfmc"]):
        ax = axes[idx]
        sub = emodb[(emodb["feature"] == feat) & (emodb["setup"] == "none")]
        if sub.empty:
            continue
        sns.barplot(
            data=sub,
            x="coeffs",
            y="test_mean",
            hue="format",
            palette=palette,
            ax=ax,
            alpha=0.9
        )
        ax.set_title(f"EmoDB {feat.upper()} (No Augmentation)", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Number of Coefficients (n)", fontsize=11)
        ax.set_ylabel("Test Accuracy (Mean)" if idx == 0 else "", fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.axhline(0.869 if feat == "mfmc" else 0.85, color="#dc2626", linestyle="--", alpha=0.7, label="Paper Target (~85-87%)")
        ax.legend(title="Format", loc="lower right", framealpha=0.9)

    plt.suptitle("Task 1: Effect of Input Representation on EmoDB Test Accuracy", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_p = RESULTS_DIR / "week4_plot_input_formats.png"
    plt.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_p.name}")


def plot_dataset_comparison(df):
    """Plot Task 2: Baseline Test Accuracy by Dataset (EmoDB, CAFE, AESDD, MESD)."""
    strat = df[(df["format"] == "baseline") & (df["split_type"] == "stratified")].copy()
    if strat.empty:
        return

    # Compare 30 coefficients for MFCC and MFMC side-by-side
    sub = strat[strat["coeffs"] == 30].copy()
    if sub.empty:
        sub = strat

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    palette = {"none": "#64748b", "version_a": "#2563eb"}

    for idx, feat in enumerate(["mfcc", "mfmc"]):
        ax = axes[idx]
        feat_sub = sub[sub["feature"] == feat]
        if feat_sub.empty:
            continue
        sns.barplot(
            data=feat_sub,
            x="dataset",
            y="test_mean",
            hue="setup",
            palette=palette,
            ax=ax,
            alpha=0.9
        )
        ax.set_title(f"30 Coeffs {feat.upper()} (Stratified 60/20/20)", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Dataset", fontsize=11)
        ax.set_ylabel("Test Accuracy (Mean)" if idx == 0 else "", fontsize=11)
        ax.set_ylim(0, 1.05)
        datasets = [d for d in feat_sub["dataset"].unique()]
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels([d.upper() for d in datasets])
        ax.legend(title="Setup", labels=["No Augmentation", "Version A (Clean 5x)"], loc="lower right")

    plt.suptitle("Task 2: Multi-Corpus Emotion Recognition Baseline (EmoDB, CAFE, AESDD, MESD)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_p = RESULTS_DIR / "week4_plot_datasets.png"
    plt.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_p.name}")


def plot_speaker_independent_comparison(df):
    """Plot Task 3: Stratified vs Speaker-Independent Split."""
    spk_data = df[(df["format"] == "baseline") & (df["setup"] == "none")].copy()
    if spk_data.empty:
        return

    # Compare on datasets with speaker splits (30 coeffs)
    sub = spk_data[spk_data["dataset"].isin(["emodb", "cafe", "aesdd"]) & (spk_data["coeffs"] == 30)].copy()
    if sub.empty:
        sub = spk_data[spk_data["dataset"].isin(["emodb", "cafe", "aesdd"])]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    palette = {"stratified": "#2563eb", "speaker_independent": "#dc2626"}

    for idx, feat in enumerate(["mfcc", "mfmc"]):
        ax = axes[idx]
        feat_sub = sub[sub["feature"] == feat]
        if feat_sub.empty:
            continue
        sns.barplot(
            data=feat_sub,
            x="dataset",
            y="test_mean",
            hue="split_type",
            palette=palette,
            ax=ax,
            alpha=0.9
        )
        ax.set_title(f"30 Coeffs {feat.upper()} (No Augmentation)", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Dataset", fontsize=11)
        ax.set_ylabel("Test Accuracy (Mean)" if idx == 0 else "", fontsize=11)
        ax.set_ylim(0, 1.0)
        datasets = [d for d in feat_sub["dataset"].unique()]
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels([d.upper() for d in datasets])
        ax.legend(title="Split Type", labels=["Stratified (Random)", "Speaker-Independent (Strict)"], loc="upper right")

    plt.suptitle("Task 3: Speaker-Independent Generalization Drop (Strict Disjointness)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_p = RESULTS_DIR / "week4_plot_speaker_independent.png"
    plt.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_p.name}")


def generate_all_summaries():
    agg = load_and_aggregate_results()
    if agg is not None:
        plot_input_format_comparison(agg)
        plot_dataset_comparison(agg)
        plot_speaker_independent_comparison(agg)


if __name__ == "__main__":
    generate_all_summaries()
