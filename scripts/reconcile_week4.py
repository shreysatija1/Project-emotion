import pandas as pd
import numpy as np

# Load week4_results.csv
df = pd.read_csv("results/week4_results.csv")
print(f"Total rows in week4_results.csv: {len(df)}")

# 1. Speaker-independent drop range
# Task 2: stratified, baseline, coeffs in [24, 30]
t2 = df[(df["split_type"] == "stratified") & (df["format"] == "baseline") & (df["coeffs"].isin([24, 30]))]
t2_summary = t2.groupby(["dataset", "feature", "coeffs", "setup"]).agg(
    strat_mean=("test_acc", "mean"),
    strat_std=("test_acc", lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0)
).reset_index()

# Task 3: speaker_independent, baseline, coeffs in [24, 30]
# per-fold aggregation first
t3 = df[df["split_type"] == "speaker_independent"]
t3_fold = t3.groupby(["dataset", "feature", "coeffs", "setup", "fold"]).agg(
    test_acc=("test_acc", "mean")
).reset_index()
t3_summary = t3_fold.groupby(["dataset", "feature", "coeffs", "setup"]).agg(
    spk_mean=("test_acc", "mean"),
    spk_std=("test_acc", lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0)
).reset_index()

merged = pd.merge(t2_summary, t3_summary, on=["dataset", "feature", "coeffs", "setup"])
merged["drop_pct_points"] = (merged["strat_mean"] - merged["spk_mean"]) * 100

print("\n--- Speaker-Independent Drops ---")
print(f"Min drop: {merged['drop_pct_points'].min():.1f} percentage points ({merged.loc[merged['drop_pct_points'].idxmin()]['dataset']} {merged.loc[merged['drop_pct_points'].idxmin()]['feature']} {merged.loc[merged['drop_pct_points'].idxmin()]['coeffs']} {merged.loc[merged['drop_pct_points'].idxmin()]['setup']})")
print(f"Max drop: {merged['drop_pct_points'].max():.1f} percentage points ({merged.loc[merged['drop_pct_points'].idxmax()]['dataset']} {merged.loc[merged['drop_pct_points'].idxmax()]['feature']} {merged.loc[merged['drop_pct_points'].idxmax()]['coeffs']} {merged.loc[merged['drop_pct_points'].idxmax()]['setup']})")
print(f"Average drop: {merged['drop_pct_points'].mean():.1f} percentage points")
print(f"Drop range exact: -{merged['drop_pct_points'].min():.1f} to -{merged['drop_pct_points'].max():.1f} points (i.e. 8.2 to 21.2 percentage points drop)")

# Per-dataset drop ranges:
for d in ["emodb", "aesdd", "cafe"]:
    sub = merged[merged["dataset"] == d]
    print(f"  {d.upper()}: drop range {sub['drop_pct_points'].min():.1f} to {sub['drop_pct_points'].max():.1f} points (mean: {sub['drop_pct_points'].mean():.1f} points)")

# 2. Augmentation shift range (Version A vs None)
# Let's check both stratified (Task 2) and speaker-independent (Task 3)
print("\n--- Augmentation Shifts (Version A minus None) ---")
# Stratified baseline:
t2_piv = t2_summary.pivot(index=["dataset", "feature", "coeffs"], columns="setup", values="strat_mean").reset_index()
t2_piv["shift"] = (t2_piv["version_a"] - t2_piv["none"]) * 100
print("Stratified baseline (Task 2, 24 & 30 coeffs):")
for idx, r in t2_piv.iterrows():
    print(f"  {r['dataset']:6s} {r['feature']:4s} {r['coeffs']:2d}: None={r['none']*100:.1f}%, VerA={r['version_a']*100:.1f}%, Shift={r['shift']:+.1f}%")
print(f"Stratified shift range: {t2_piv['shift'].min():+.1f}% to {t2_piv['shift'].max():+.1f}% (mean: {t2_piv['shift'].mean():+.1f}%)")

# Also check 12 coeffs stratified on EmoDB if in df:
t2_all = df[(df["split_type"] == "stratified") & (df["format"] == "baseline")]
t2_all_summary = t2_all.groupby(["dataset", "feature", "coeffs", "setup"])["test_acc"].mean().unstack().reset_index()
if "none" in t2_all_summary.columns and "version_a" in t2_all_summary.columns:
    t2_all_summary["shift"] = (t2_all_summary["version_a"] - t2_all_summary["none"]) * 100
    print(f"All stratified baseline shift range (including 12 coeffs): {t2_all_summary['shift'].min():+.1f}% to {t2_all_summary['shift'].max():+.1f}%")

# Speaker-independent (Task 3):
t3_piv = t3_summary.pivot(index=["dataset", "feature", "coeffs"], columns="setup", values="spk_mean").reset_index()
t3_piv["shift"] = (t3_piv["version_a"] - t3_piv["none"]) * 100
print("\nSpeaker-independent (Task 3, 24 & 30 coeffs):")
for idx, r in t3_piv.iterrows():
    print(f"  {r['dataset']:6s} {r['feature']:4s} {r['coeffs']:2d}: None={r['none']*100:.1f}%, VerA={r['version_a']*100:.1f}%, Shift={r['shift']:+.1f}%")
print(f"Speaker-independent shift range: {t3_piv['shift'].min():+.1f}% to {t3_piv['shift'].max():+.1f}% (mean: {t3_piv['shift'].mean():+.1f}%)")

# Overall augmentation shift across both split types:
all_shifts = pd.concat([t2_piv["shift"], t3_piv["shift"]])
print(f"Overall shift range across stratified & spk-indep: {all_shifts.min():+.1f}% to {all_shifts.max():+.1f}%")

# Let's inspect where -7 to +11, -1.5 to +3.5, and -2 to +5 came from:
print(f"  Task 2 stratified shift range: {t2_piv['shift'].min():+.1f} to {t2_piv['shift'].max():+.1f} (min is CAFE mfcc 30: -6.2%, max is AESDD mfmc 24: +8.3%, or if looking at individual seed runs...)")

# Let's check slide 10's speaker independent accuracy ranges:
print("\n--- Speaker-Independent Accuracy Ranges by Dataset ---")
for d in ["emodb", "aesdd", "cafe"]:
    sub = t3_summary[t3_summary["dataset"] == d]
    print(f"  {d.upper()}: min {sub['spk_mean'].min()*100:.1f}%, max {sub['spk_mean'].max()*100:.1f}%")
