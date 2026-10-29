import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results")
df = pd.read_csv(RESULTS_DIR / "week5_results.csv")
t1 = df[df["task"] == "task1_regularization"]

print("=== TASK 1 DETAILED RESULTS ===")

# Stratified
strat = t1[t1["split_type"] == "stratified"]
strat_summary = strat.groupby("reg_config").agg(
    val_acc=("val_acc", "mean"),
    val_std=("val_acc", lambda x: np.std(x, ddof=1)),
    test_acc=("test_acc", "mean"),
    test_std=("test_acc", lambda x: np.std(x, ddof=1)),
    train_acc=("train_acc", "mean"),
    gap=("train_val_gap", "mean"),
    gap_std=("train_val_gap", lambda x: np.std(x, ddof=1)),
).reset_index()

print("\n--- STRATIFIED SPLIT (EmoDB, N=12 per config) ---")
for _, r in strat_summary.sort_values(by="val_acc", ascending=False).iterrows():
    print(f"{r['reg_config']:20s} | Val: {r['val_acc']*100:.1f}±{r['val_std']*100:.1f}% | Test: {r['test_acc']*100:.1f}±{r['test_std']*100:.1f}% | Train-Val Gap: {r['gap']*100:.1f}±{r['gap_std']*100:.1f}%")

# Speaker Independent (per fold average first, then mean ± std across folds)
spk = t1[t1["split_type"] == "speaker_independent"]
fold_agg = spk.groupby(["reg_config", "feature", "coeffs", "fold"]).agg(
    val_acc=("val_acc", "mean"),
    test_acc=("test_acc", "mean"),
    train_acc=("train_acc", "mean"),
    gap=("train_val_gap", "mean"),
).reset_index()

spk_summary = fold_agg.groupby("reg_config").agg(
    val_acc=("val_acc", "mean"),
    val_std=("val_acc", lambda x: np.std(x, ddof=1)),
    test_acc=("test_acc", "mean"),
    test_std=("test_acc", lambda x: np.std(x, ddof=1)),
    train_acc=("train_acc", "mean"),
    gap=("gap", "mean"),
    gap_std=("gap", lambda x: np.std(x, ddof=1)),
).reset_index()

print("\n--- SPEAKER-INDEPENDENT FOLDS (EmoDB, 5 folds, N=20 fold-conditions per config) ---")
for _, r in spk_summary.sort_values(by="val_acc", ascending=False).iterrows():
    print(f"{r['reg_config']:20s} | Val: {r['val_acc']*100:.1f}±{r['val_std']*100:.1f}% | Test: {r['test_acc']*100:.1f}±{r['test_std']*100:.1f}% | Train-Val Gap: {r['gap']*100:.1f}±{r['gap_std']*100:.1f}%")

# Combined comparison table
merged = pd.merge(strat_summary, spk_summary, on="reg_config", suffixes=("_strat", "_spk"))
merged.to_csv(RESULTS_DIR / "task1_summary_table.csv", index=False)
print("\nSaved summary table to results/task1_summary_table.csv")
