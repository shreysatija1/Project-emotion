"""
Task 3: Statistical Comparison Module for Week 5.

Performs paired statistical hypothesis testing and computes confidence intervals for:
1. Baseline vs. Best Regularized Model (EmoDB: Stratified and Speaker-Independent)
2. None vs. Each Acoustic Augmentation (Speaker-Independent across EmoDB, AESDD, CAFE):
   - None vs. Pitch Shift (±2 semitones)
   - None vs. Time Stretch (0.9x-1.1x)
   - None vs. SpecAugment (time & frequency masking)
   - None vs. Version A (Gaussian Noise)

Outputs:
- Mean difference (percentage points)
- 95% Confidence Interval [CI_lower, CI_upper]
- Paired t-statistic and two-tailed p-value
- Plain-language verdict on whether difference is distinguishable from noise (p < 0.05).
- Saves formatted tables to results/week5_statistical_comparison.csv and markdown.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
WEEK4_CSV = RESULTS_DIR / "week4_results.csv"
WEEK5_CSV = RESULTS_DIR / "week5_results.csv"


def compute_paired_stats(diffs, alpha=0.05):
    """
    Given array of paired differences, return summary statistics.
    """
    diffs = np.asarray(diffs, dtype=np.float64)
    n = len(diffs)
    if n < 2:
        return {
            "n": n,
            "mean_diff": np.nan,
            "std_diff": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "verdict": "insufficient_data"
        }
    
    mean_d = np.mean(diffs)
    std_d = np.std(diffs, ddof=1)
    se_d = std_d / np.sqrt(n)
    
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
    ci_low = mean_d - t_crit * se_d
    ci_high = mean_d + t_crit * se_d
    
    t_stat, p_val = stats.ttest_1samp(diffs, popmean=0.0)
    
    sig = p_val < alpha
    if sig:
        direction = "gain" if mean_d > 0 else "loss"
        verdict = f"Statistically significant {direction} (p = {p_val:.4f})"
    else:
        verdict = f"Indistinguishable from noise (p = {p_val:.4f})"
        
    return {
        "n": n,
        "mean_diff_pct": round(mean_d * 100, 2),
        "std_diff_pct": round(std_d * 100, 2),
        "ci_lower_pct": round(ci_low * 100, 2),
        "ci_upper_pct": round(ci_high * 100, 2),
        "t_stat": round(t_stat, 3),
        "p_value": round(p_val, 4),
        "verdict": verdict
    }


def analyze_regularization(w5_df):
    """
    Compare baseline vs each regularized configuration on EmoDB.
    """
    t1 = w5_df[w5_df["task"] == "task1_regularization"].copy()
    if t1.empty:
        return pd.DataFrame()
        
    records = []
    for split_type in ["stratified", "speaker_independent"]:
        sub_split = t1[t1["split_type"] == split_type]
        base = sub_split[sub_split["reg_config"] == "reg_none"]
        
        reg_configs = [c for c in sub_split["reg_config"].unique() if c != "reg_none"]
        for reg in reg_configs:
            comp = sub_split[sub_split["reg_config"] == reg]
            
            # Match on feature, coeffs, fold, seed
            join_cols = ["feature", "coeffs", "fold", "seed"] if split_type == "speaker_independent" else ["feature", "coeffs", "seed"]
            merged = pd.merge(base, comp, on=join_cols, suffixes=("_base", "_reg"))
            
            # Val accuracy diff
            diff_val = (merged["val_acc_reg"] - merged["val_acc_base"]).values
            st_val = compute_paired_stats(diff_val)
            
            # Test accuracy diff
            diff_test = (merged["test_acc_reg"] - merged["test_acc_base"]).values
            st_test = compute_paired_stats(diff_test)
            
            # Train-val gap diff (negative diff means gap narrowed)
            diff_gap = (merged["train_val_gap_reg"] - merged["train_val_gap_base"]).values
            st_gap = compute_paired_stats(diff_gap)
            
            records.append({
                "comparison": f"Baseline vs {reg}",
                "split_type": split_type,
                "n_pairs": st_test["n"],
                "val_diff_pct": st_val["mean_diff_pct"],
                "val_p_value": st_val["p_value"],
                "test_diff_pct": st_test["mean_diff_pct"],
                "test_ci_95": f"[{st_test['ci_lower_pct']:+.1f}%, {st_test['ci_upper_pct']:+.1f}%]",
                "test_p_value": st_test["p_value"],
                "gap_diff_pct": st_gap["mean_diff_pct"],
                "gap_p_value": st_gap["p_value"],
                "test_verdict": st_test["verdict"]
            })
            
    df_out = pd.DataFrame(records)
    return df_out


def analyze_augmentations(w4_df, w5_df):
    """
    Compare None vs each augmentation under speaker-disjoint folds.
    """
    # Baseline unaugmented from Week 4 task 3
    t3_none = w4_df[(w4_df["split_type"] == "speaker_independent") & (w4_df["setup"] == "none")].copy()
    t3_vera = w4_df[(w4_df["split_type"] == "speaker_independent") & (w4_df["setup"] == "version_a")].copy()
    
    t2_w5 = w5_df[w5_df["task"] == "task2_augmentation"].copy()
    if t2_w5.empty:
        return pd.DataFrame()
        
    records = []
    
    # 1. Compare None vs Version A (from Week 4)
    join_cols = ["dataset", "feature", "coeffs", "fold", "seed"]
    m_vera = pd.merge(t3_none, t3_vera, on=join_cols, suffixes=("_none", "_aug"))
    for d in ["all", "emodb", "aesdd", "cafe"]:
        sub = m_vera if d == "all" else m_vera[m_vera["dataset"] == d]
        diffs = (sub["test_acc_aug"] - sub["test_acc_none"]).values
        st = compute_paired_stats(diffs)
        records.append({
            "dataset": d.upper(),
            "augmentation": "version_a (Gaussian noise)",
            "n_pairs": st["n"],
            "mean_diff_pct": st["mean_diff_pct"],
            "ci_95": f"[{st['ci_lower_pct']:+.1f}%, {st['ci_upper_pct']:+.1f}%]",
            "t_stat": st["t_stat"],
            "p_value": st["p_value"],
            "verdict": st["verdict"]
        })
        
    # 2. Compare None vs Week 5 Augmentations (pitch_shift, time_stretch, spec_augment)
    for aug in ["pitch_shift", "time_stretch", "spec_augment"]:
        sub_aug = t2_w5[t2_w5["setup"] == aug]
        merged = pd.merge(t3_none, sub_aug, on=join_cols, suffixes=("_none", "_aug"))
        for d in ["all", "emodb", "aesdd", "cafe"]:
            sub = merged if d == "all" else merged[merged["dataset"] == d]
            if len(sub) == 0:
                continue
            diffs = (sub["test_acc_aug"] - sub["test_acc_none"]).values
            st = compute_paired_stats(diffs)
            records.append({
                "dataset": d.upper(),
                "augmentation": aug,
                "n_pairs": st["n"],
                "mean_diff_pct": st["mean_diff_pct"],
                "ci_95": f"[{st['ci_lower_pct']:+.1f}%, {st['ci_upper_pct']:+.1f}%]",
                "t_stat": st["t_stat"],
                "p_value": st["p_value"],
                "verdict": st["verdict"]
            })
            
    df_out = pd.DataFrame(records)
    return df_out


def run_statistical_analysis():
    if not WEEK5_CSV.is_file():
        print("week5_results.csv does not exist yet.")
        return
        
    w5_df = pd.read_csv(WEEK5_CSV)
    w4_df = pd.read_csv(WEEK4_CSV)
    
    print("\n================== TASK 1: REGULARIZATION STATISTICAL COMPARISON ==================")
    df_reg = analyze_regularization(w5_df)
    if not df_reg.empty:
        print(df_reg.to_string(index=False))
        out_reg = RESULTS_DIR / "week5_task1_statistical_comparison.csv"
        df_reg.to_csv(out_reg, index=False)
        print(f"\nSaved Task 1 statistical table to {out_reg}")
        
    print("\n================== TASK 2: ACOUSTIC AUGMENTATION STATISTICAL COMPARISON ==================")
    df_aug = analyze_augmentations(w4_df, w5_df)
    if not df_aug.empty:
        print(df_aug.to_string(index=False))
        out_aug = RESULTS_DIR / "week5_task2_statistical_comparison.csv"
        df_aug.to_csv(out_aug, index=False)
        print(f"\nSaved Task 2 statistical table to {out_aug}")


if __name__ == "__main__":
    run_statistical_analysis()
