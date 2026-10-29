"""
Task 3 Metric Aggregator and Comparator.

In accordance with supervisor instructions:
1. Average seeds within each fold:
   fold_test_acc = mean(test_acc across seeds 42, 43, 44)
2. Mean ± Std across folds:
   mean_acc = mean(fold_test_acc across folds)
   std_acc = std(fold_test_acc across folds, ddof=1)
3. Save per-fold and per-speaker results in CSV:
   results/week4_task3_per_fold.csv
   results/week4_task3_summary.csv
4. Generate comparison table and plots comparing stratified vs. speaker-independent.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
CSV_FILE = RESULTS_DIR / "week4_results.csv"


def aggregate_task3():
    if not CSV_FILE.is_file():
        print(f"Error: {CSV_FILE} does not exist.")
        return

    df = pd.read_csv(CSV_FILE)
    t3 = df[df["split_type"] == "speaker_independent"].copy()

    if t3.empty:
        print("No speaker-independent runs found in week4_results.csv yet.")
        return

    print(f"Found {len(t3)} speaker-independent runs.")

    # 1. Per-fold aggregation (averaging across seeds within each fold)
    fold_grp = ["dataset", "feature", "coeffs", "setup", "fold"]
    per_fold = t3.groupby(fold_grp).agg(
        test_acc=("test_acc", "mean"),
        val_acc=("val_acc", "mean"),
        train_acc=("train_acc", "mean"),
        test_loss=("test_loss", "mean"),
        val_loss=("val_loss", "mean"),
        test_spk=("test_spk", "first"),
        val_spk=("val_spk", "first"),
        seed_count=("seed", "count"),
        avg_duration=("duration_s", "mean"),
    ).reset_index()

    per_fold_out = RESULTS_DIR / "week4_task3_per_fold.csv"
    per_fold.to_csv(per_fold_out, index=False)
    print(f"Saved per-fold results to {per_fold_out}")

    # 2. Across-fold aggregation (mean ± std across folds)
    summary_grp = ["dataset", "feature", "coeffs", "setup"]
    summary = per_fold.groupby(summary_grp).agg(
        test_mean=("test_acc", "mean"),
        test_std=("test_acc", lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0),
        val_mean=("val_acc", "mean"),
        val_std=("val_acc", lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0),
        train_mean=("train_acc", "mean"),
        train_std=("train_acc", lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0),
        n_folds=("fold", "count"),
    ).reset_index()

    # Format percentages for clean presentation
    summary["test_acc_formatted"] = summary.apply(
        lambda r: f"{r['test_mean']*100:.1f}% ± {r['test_std']*100:.1f}%", axis=1
    )
    summary["val_acc_formatted"] = summary.apply(
        lambda r: f"{r['val_mean']*100:.1f}% ± {r['val_std']*100:.1f}%", axis=1
    )

    summary_out = RESULTS_DIR / "week4_task3_summary.csv"
    summary.to_csv(summary_out, index=False)
    print(f"Saved Task 3 cross-fold summary to {summary_out}")

    # 3. Build Comparison Table with Stratified Baseline (Task 2)
    strat = df[(df["split_type"] == "stratified") & (df["format"] == "baseline")].copy()
    strat_summary = strat.groupby(["dataset", "feature", "coeffs", "setup"]).agg(
        strat_test_mean=("test_acc", "mean"),
        strat_test_std=("test_acc", lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0),
    ).reset_index()

    comparison = pd.merge(strat_summary, summary, on=["dataset", "feature", "coeffs", "setup"], how="inner")
    comparison["drop_mean"] = comparison["strat_test_mean"] - comparison["test_mean"]
    comparison["drop_formatted"] = comparison.apply(
        lambda r: f"-{r['drop_mean']*100:.1f}%", axis=1
    )
    comparison["strat_formatted"] = comparison.apply(
        lambda r: f"{r['strat_test_mean']*100:.1f}% ± {r['strat_test_std']*100:.1f}%", axis=1
    )

    comp_out = RESULTS_DIR / "week4_comparison_stratified_vs_spk_independent.csv"
    comparison.to_csv(comp_out, index=False)
    print(f"Saved stratified vs speaker-independent comparison to {comp_out}")

    # Display comparison
    print("\n" + "="*80)
    print("STRATIFIED VS. SPEAKER-INDEPENDENT TEST ACCURACY COMPARISON")
    print("="*80)
    for _, row in comparison.iterrows():
        print(
            f"{row['dataset'].upper():<7} | {row['feature'].upper():<4} {row['coeffs']} | {row['setup']:<9} | "
            f"Strat: {row['strat_formatted']:<16} | Spk-Indep: {row['test_acc_formatted']:<16} | Drop: {row['drop_formatted']}"
        )
    print("="*80)

    # 4. Generate comparison plots
    plot_speaker_independent_comparison(comparison)


def plot_speaker_independent_comparison(comp_df):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"

    # Focus on 30 coeffs none vs version_a
    sub = comp_df[comp_df["coeffs"] == 30].copy()
    if sub.empty:
        sub = comp_df.copy()

    datasets = [d for d in ["emodb", "aesdd", "cafe"] if d in sub["dataset"].unique()]
    if not datasets:
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    for idx, feat in enumerate(["mfcc", "mfmc"]):
        ax = axes[idx]
        feat_sub = sub[(sub["feature"] == feat) & (sub["setup"] == "none")].copy()
        if feat_sub.empty:
            continue

        x = np.arange(len(datasets))
        width = 0.35

        strat_vals = []
        strat_errs = []
        spk_vals = []
        spk_errs = []

        for d in datasets:
            match = feat_sub[feat_sub["dataset"] == d]
            if not match.empty:
                r = match.iloc[0]
                strat_vals.append(r["strat_test_mean"])
                strat_errs.append(r["strat_test_std"])
                spk_vals.append(r["test_mean"])
                spk_errs.append(r["test_std"])
            else:
                strat_vals.append(0)
                strat_errs.append(0)
                spk_vals.append(0)
                spk_errs.append(0)

        rects1 = ax.bar(x - width/2, strat_vals, width, yerr=strat_errs, capsize=4,
                        label="Stratified (Speaker Overlap)", color="#2563eb", alpha=0.9)
        rects2 = ax.bar(x + width/2, spk_vals, width, yerr=spk_errs, capsize=4,
                        label="Speaker-Independent (Zero Overlap)", color="#dc2626", alpha=0.9)

        # Annotate drops
        for i, d in enumerate(datasets):
            s_val = strat_vals[i]
            spk_val = spk_vals[i]
            if s_val > 0 and spk_val > 0:
                drop = s_val - spk_val
                ax.annotate(
                    f"Drop:\n-{drop*100:.1f}%",
                    xy=(x[i] + width/2, spk_val / 2),
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#991b1b", alpha=0.8, edgecolor="none")
                )

        ax.set_title(f"30 Coeffs {feat.upper()} (No Augmentation)", fontsize=13, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([d.upper() for d in datasets], fontsize=11, fontweight="bold")
        ax.set_ylabel("Test Accuracy" if idx == 0 else "", fontsize=11)
        ax.set_ylim(0, 1.05)

        # Chance levels
        chance_map = {"emodb": 1/7, "cafe": 1/7, "aesdd": 1/5}
        for d_idx, d in enumerate(datasets):
            ch = chance_map.get(d, 1/7)
            ax.hlines(y=ch, xmin=d_idx-0.45, xmax=d_idx+0.45, colors="#64748b", linestyles=":", linewidth=1.5)
            ax.text(d_idx, ch + 0.02, f"Chance: {ch*100:.1f}%", ha="center", fontsize=8, color="#475569")

        ax.legend(loc="upper right", framealpha=0.95)

    plt.suptitle("Task 3: Speaker-Independent Generalization Collapse Across Corpora", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_fig = RESULTS_DIR / "week4_plot_speaker_independent.png"
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to {out_fig}")


if __name__ == "__main__":
    aggregate_task3()
