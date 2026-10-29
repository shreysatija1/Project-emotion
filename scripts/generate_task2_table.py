import pandas as pd
import numpy as np
from pathlib import Path

w4 = pd.read_csv("results/week4_results.csv")
w5 = pd.read_csv("results/week5_results.csv")

t3_w4 = w4[w4["split_type"] == "speaker_independent"]
t2_w5 = w5[w5["task"] == "task2_augmentation"]

# Combine Week 4 baseline (none and version_a) with Week 5 augmentations
df_all = pd.concat([
    t3_w4[t3_w4["setup"].isin(["none", "version_a"])],
    t2_w5
], ignore_index=True)

# Protocol: average seeds (42, 43, 44) within each fold first
fold_means = df_all.groupby(["dataset", "setup", "feature", "coeffs", "fold"])["test_acc"].mean().reset_index()

# Across-fold mean and std per (dataset, setup, feature, coeffs)
config_summary = fold_means.groupby(["dataset", "setup", "feature", "coeffs"]).agg(
    test_mean=("test_acc", "mean"),
    test_std=("test_acc", lambda x: np.std(x, ddof=1)),
    n_folds=("fold", "count")
).reset_index()

# Overall per (dataset, setup) across all feature/coeff conditions
dataset_aug_summary = fold_means.groupby(["dataset", "setup"]).agg(
    test_mean=("test_acc", "mean"),
    test_std=("test_acc", lambda x: np.std(x, ddof=1)),
    n_folds=("fold", "count")
).reset_index()

print("=== TASK 2: SPEAKER-INDEPENDENT TEST ACCURACY PER AUGMENTATION ===")
print("\n--- Summary per Dataset and Augmentation (Fold-Aggregated) ---")
for d in ["emodb", "aesdd", "cafe"]:
    sub = dataset_aug_summary[dataset_aug_summary["dataset"] == d].copy()
    none_row = sub[sub["setup"] == "none"].iloc[0]
    base_acc = none_row["test_mean"]
    base_std = none_row["test_std"]
    print(f"\n{d.upper()}:")
    for _, r in sub.iterrows():
        diff = (r["test_mean"] - base_acc) * 100
        exceeds = abs(diff) > (r["test_std"] * 100)
        note = "EXCEEDS fold std" if exceeds else "within cross-fold std"
        print(f"  {r['setup']:15s} | Acc: {r['test_mean']*100:.1f}% ± {r['test_std']*100:.1f}% | Shift vs None: {diff:+.1f}% | ({note})")

# Save detailed configuration table
config_summary["formatted"] = config_summary.apply(lambda r: f"{r['test_mean']*100:.1f}% ± {r['test_std']*100:.1f}%", axis=1)
config_piv = config_summary.pivot(index=["dataset", "feature", "coeffs"], columns="setup", values="formatted").reset_index()
config_piv.to_csv("results/task2_per_augmentation_table.csv", index=False)
print("\nSaved detailed per-config table to results/task2_per_augmentation_table.csv")
