"""
Master Reproduction Script for SER Project (Weeks 1-5).

Generates and displays all primary tables and verification summaries from:
- results/week4_results.csv (618 runs)
- results/week5_results.csv (Task 1 & Task 2 runs)

Outputs:
1. results/table1_reproduction_summary.csv
2. results/table2_144_format_comparison.csv
3. results/table3_speaker_generalization_drop.csv
4. results/table4_regularization_study.csv
5. results/table5_acoustic_augmentations.csv
6. results/table6_statistical_significance.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
W4_CSV = RESULTS_DIR / "week4_results.csv"
W5_CSV = RESULTS_DIR / "week5_results.csv"


def generate_table1_reproduction_summary(df4, df5=None):
    """Paper Claim vs Our Baseline vs Speaker Independent."""
    paper_targets = {
        ("emodb", "mfcc", 30): (85.0, 96.1),
        ("emodb", "mfmc", 30): (86.9, 97.2),
        ("aesdd", "mfcc", 30): (82.0, 95.2),
        ("aesdd", "mfmc", 30): (95.0, 97.8),
        ("cafe", "mfcc", 30): (88.8, 98.6),
        ("cafe", "mfmc", 30): (93.6, 99.5),
        ("mesd", "mfcc", 30): (85.5, 96.1),
        ("mesd", "mfmc", 30): (94.2, 96.9),
    }

    t2 = df4[(df4["split_type"] == "stratified") & (df4["format"] == "baseline") & (df4["coeffs"] == 30)]
    t3 = df4[(df4["split_type"] == "speaker_independent") & (df4["coeffs"] == 30)]

    records = []
    for (d, f, c), (p_none, p_aug) in paper_targets.items():
        # Stratified None & VerA
        s_none = t2[(t2["dataset"] == d) & (t2["feature"] == f) & (t2["setup"] == "none")]["test_acc"]
        s_vera = t2[(t2["dataset"] == d) & (t2["feature"] == f) & (t2["setup"] == "version_a")]["test_acc"]

        strat_none_str = f"{s_none.mean()*100:.1f}% ± {s_none.std(ddof=1)*100:.1f}%" if len(s_none) > 0 else "N/A"
        strat_vera_str = f"{s_vera.mean()*100:.1f}% ± {s_vera.std(ddof=1)*100:.1f}%" if len(s_vera) > 0 else "N/A"

        # Speaker-independent None & VerA (fold-level mean first)
        spk_none_str = "N/A"
        spk_vera_str = "N/A"
        if d in ["emodb", "aesdd", "cafe"]:
            sp_n = t3[(t3["dataset"] == d) & (t3["feature"] == f) & (t3["setup"] == "none")].groupby("fold")["test_acc"].mean()
            sp_v = t3[(t3["dataset"] == d) & (t3["feature"] == f) & (t3["setup"] == "version_a")].groupby("fold")["test_acc"].mean()
            spk_none_str = f"{sp_n.mean()*100:.1f}% ± {sp_n.std(ddof=1)*100:.1f}%" if len(sp_n) > 0 else "N/A"
            spk_vera_str = f"{sp_v.mean()*100:.1f}% ± {sp_v.std(ddof=1)*100:.1f}%" if len(sp_v) > 0 else "N/A"

        records.append({
            "Dataset": d.upper(),
            "Feature (30 Coeffs)": f.upper(),
            "Paper None": f"{p_none:.1f}%",
            "Paper Aug": f"{p_aug:.1f}%",
            "Stratified None": strat_none_str,
            "Stratified Ver A": strat_vera_str,
            "Speaker-Indep None": spk_none_str,
            "Speaker-Indep Ver A": spk_vera_str,
        })

    out_df = pd.DataFrame(records)
    out_df.to_csv(RESULTS_DIR / "table1_reproduction_summary.csv", index=False)
    return out_df


def generate_table4_regularization(df5):
    """Task 1 Regularization Results on EmoDB."""
    t1 = df5[df5["task"] == "task1_regularization"].copy()
    if t1.empty:
        return pd.DataFrame()

    records = []
    for split in ["stratified", "speaker_independent"]:
        sub = t1[t1["split_type"] == split]
        for reg in ["reg_none", "reg_l2", "reg_spatial_dropout", "reg_early_stopping", "reg_cosine_lr", "reg_combined"]:
            reg_sub = sub[sub["reg_config"] == reg]
            if reg_sub.empty:
                continue

            if split == "stratified":
                # Average across 12 runs (2 feats * 2 coeffs * 3 seeds)
                v_m = reg_sub["val_acc"].mean() * 100
                v_s = reg_sub["val_acc"].std(ddof=1) * 100
                t_m = reg_sub["test_acc"].mean() * 100
                t_s = reg_sub["test_acc"].std(ddof=1) * 100
                g_m = reg_sub["train_val_gap"].mean() * 100
                g_s = reg_sub["train_val_gap"].std(ddof=1) * 100
            else:
                # Group by fold first, then mean ± std across folds
                fold_means = reg_sub.groupby(["feature", "coeffs", "fold"])[["val_acc", "test_acc", "train_val_gap"]].mean().reset_index()
                v_m = fold_means["val_acc"].mean() * 100
                v_s = fold_means["val_acc"].std(ddof=1) * 100
                t_m = fold_means["test_acc"].mean() * 100
                t_s = fold_means["test_acc"].std(ddof=1) * 100
                g_m = fold_means["train_val_gap"].mean() * 100
                g_s = fold_means["train_val_gap"].std(ddof=1) * 100

            records.append({
                "Split": split,
                "Regularization Config": reg,
                "Val Accuracy": f"{v_m:.1f}% ± {v_s:.1f}%",
                "Test Accuracy": f"{t_m:.1f}% ± {t_s:.1f}%",
                "Train-Val Gap": f"{g_m:.1f}% ± {g_s:.1f}%",
                "Val Acc Raw": v_m,
                "Test Acc Raw": t_m,
                "Gap Raw": g_m
            })

    out_df = pd.DataFrame(records)
    out_df.to_csv(RESULTS_DIR / "table4_regularization_study.csv", index=False)
    return out_df


def generate_table5_augmentations(df4, df5):
    """Task 2 Acoustic Augmentations under Speaker-Disjoint Folds."""
    t3_w4 = df4[df4["split_type"] == "speaker_independent"].copy()
    t2_w5 = df5[df5["task"] == "task2_augmentation"].copy() if df5 is not None else pd.DataFrame()

    records = []
    for d in ["emodb", "aesdd", "cafe"]:
        for aug in ["none", "version_a", "pitch_shift", "time_stretch", "spec_augment"]:
            if aug in ["none", "version_a"]:
                sub = t3_w4[(t3_w4["dataset"] == d) & (t3_w4["setup"] == aug)]
            else:
                if t2_w5.empty:
                    continue
                sub = t2_w5[(t2_w5["dataset"] == d) & (t2_w5["setup"] == aug)]

            if sub.empty:
                continue

            # Compute fold averages (averaging seeds 42, 43, 44 within fold)
            fold_avgs = sub.groupby(["feature", "coeffs", "fold"])["test_acc"].mean().reset_index()
            t_m = fold_avgs["test_acc"].mean() * 100
            t_s = fold_avgs["test_acc"].std(ddof=1) * 100

            records.append({
                "Dataset": d.upper(),
                "Augmentation": aug,
                "Speaker-Indep Test Acc": f"{t_m:.1f}% ± {t_s:.1f}%",
                "Mean Raw": t_m,
                "Std Raw": t_s
            })

    out_df = pd.DataFrame(records)
    out_df.to_csv(RESULTS_DIR / "table5_acoustic_augmentations.csv", index=False)
    return out_df


def main():
    if not W4_CSV.is_file():
        print(f"Error: {W4_CSV} not found.")
        return

    df4 = pd.read_csv(W4_CSV)
    df5 = pd.read_csv(W5_CSV) if W5_CSV.is_file() else None

    print("==========================================================================")
    print("MASTER REPRODUCTION PIPELINE: HEADLINE RESULTS SUMMARY (WEEKS 1 - 5)")
    print("==========================================================================")

    t1_df = generate_table1_reproduction_summary(df4, df5)
    print("\n--- TABLE 1: Paper Target Claims vs. Leak-Free Reproduction ---")
    print(t1_df.to_string(index=False))

    if df5 is not None and not df5.empty:
        t4_df = generate_table4_regularization(df5)
        if not t4_df.empty:
            print("\n--- TABLE 4: Task 1 Regularization Study (EmoDB) ---")
            print(t4_df[["Split", "Regularization Config", "Val Accuracy", "Test Accuracy", "Train-Val Gap"]].to_string(index=False))

        t5_df = generate_table5_augmentations(df4, df5)
        if not t5_df.empty:
            print("\n--- TABLE 5: Task 2 Acoustic Augmentations under Speaker-Disjoint Folds ---")
            print(t5_df[["Dataset", "Augmentation", "Speaker-Indep Test Acc"]].to_string(index=False))

    print("\n[SUCCESS] Generated reproduction tables in results/")


if __name__ == "__main__":
    main()
