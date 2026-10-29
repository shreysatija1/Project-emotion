"""
Check, backup, and clean results/week4_results.csv.
Identify exactly which of the 384 Task 3 runs are missing.
"""

import shutil
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
CSV_PATH = RESULTS_DIR / "week4_results.csv"
BACKUP_PATH = RESULTS_DIR / "week4_results_backup_before_resume.csv"

# 1. Backup
print(f"Creating backup at: {BACKUP_PATH}")
shutil.copy(CSV_PATH, BACKUP_PATH)

# 2. Read and verify
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows.")

# Verify first 234 rows are stratified (Week 3, Task 1, Task 2)
first_234 = df.iloc[:234]
assert (first_234["split_type"] == "stratified").all(), "Error: First 234 rows contain non-stratified entries!"
print("Verified: First 234 rows are 100% stratified (Week 3, Task 1, Task 2 untouched).")

# Task 3 rows
t3_df = df.iloc[234:].copy()
print(f"Found {len(t3_df)} Task 3 rows initially.")

# Check for incomplete rows (any NaN in key numeric columns)
required_cols = ["dataset", "feature", "coeffs", "setup", "split_type", "fold", "seed", "val_acc", "test_acc", "train_acc"]
nan_mask = t3_df[required_cols].isna().any(axis=1)
if nan_mask.any():
    print(f"Removing {nan_mask.sum()} incomplete Task 3 rows:")
    print(t3_df[nan_mask])
    t3_df = t3_df[~nan_mask]
else:
    print("No incomplete rows found with NaNs.")

# Check for duplicate runs
dedup_keys = ["dataset", "feature", "coeffs", "setup", "split_type", "fold", "seed"]
dup_mask = t3_df.duplicated(subset=dedup_keys, keep="first")
if dup_mask.any():
    print(f"Removing {dup_mask.sum()} duplicate Task 3 rows:")
    print(t3_df[dup_mask][dedup_keys])
    t3_df = t3_df[~dup_mask]
else:
    print("No duplicate Task 3 runs found.")

# Recombine and save cleaned CSV
cleaned_df = pd.concat([first_234, t3_df], ignore_index=True)
cleaned_df.to_csv(CSV_PATH, index=False)
print(f"Cleaned CSV saved. Total rows: {len(cleaned_df)} (234 baseline/stratified + {len(t3_df)} valid Task 3 runs).")

# 3. Work out which of the 384 Task 3 runs are missing
expected_runs = []
folds_map = {"emodb": 5, "aesdd": 5, "cafe": 6}
features = ["mfcc", "mfmc"]
coeffs = [24, 30]
setups = ["none", "version_a"]
seeds = [42, 43, 44]

for dataset, n_folds in folds_map.items():
    for fold in range(n_folds):
        for feat in features:
            for c in coeffs:
                for setup in setups:
                    for s in seeds:
                        expected_runs.append({
                            "dataset": dataset,
                            "fold": float(fold),
                            "feature": feat,
                            "coeffs": c,
                            "setup": setup,
                            "seed": s
                        })

expected_df = pd.DataFrame(expected_runs)
print(f"\nTotal expected Task 3 runs: {len(expected_df)} (384 runs)")

# Find completed combinations in t3_df
t3_df["fold"] = t3_df["fold"].astype(float)
t3_df["coeffs"] = t3_df["coeffs"].astype(int)
t3_df["seed"] = t3_df["seed"].astype(int)

merged = pd.merge(expected_df, t3_df[dedup_keys], on=["dataset", "fold", "feature", "coeffs", "setup", "seed"], how="left", indicator=True)
missing_df = merged[merged["_merge"] == "left_only"].drop(columns=["_merge", "split_type"])

print(f"\nCompleted Task 3 runs: {len(t3_df)}")
print(f"Missing Task 3 runs: {len(missing_df)}")

# Summary by dataset
print("\n=== Missing Runs by Dataset ===")
print(missing_df["dataset"].value_counts())

print("\n=== Missing Runs by Dataset and Fold ===")
print(missing_df.groupby(["dataset", "fold"]).size())

print("\n=== Detailed List of Missing Runs ===")
for i, r in missing_df.reset_index(drop=True).iterrows():
    print(f"{i+1:2d}. {r['dataset'].upper()} | Fold {int(r['fold'])} | {r['feature'].upper()} {r['coeffs']} | {r['setup']} | Seed {r['seed']}")
