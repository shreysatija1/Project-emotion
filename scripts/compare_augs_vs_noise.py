import pandas as pd
import numpy as np
from scipy import stats

w4 = pd.read_csv("results/week4_results.csv")
w5 = pd.read_csv("results/week5_results.csv")

t3_w4 = w4[w4["split_type"] == "speaker_independent"]
t3_vera = t3_w4[t3_w4["setup"] == "version_a"]
t2_w5 = w5[w5["task"] == "task2_augmentation"]

join_cols = ["dataset", "feature", "coeffs", "fold", "seed"]

print("=== DIRECT COMPARISON: NOVEL AUGS VS. GAUSSIAN NOISE (VERSION A) ===")
for aug in ["pitch_shift", "time_stretch", "spec_augment"]:
    sub_aug = t2_w5[t2_w5["setup"] == aug]
    merged = pd.merge(sub_aug, t3_vera, on=join_cols, suffixes=("_new", "_vera"))
    
    diff = (merged["test_acc_new"] - merged["test_acc_vera"]).values
    n = len(diff)
    mean_d = np.mean(diff) * 100
    std_d = np.std(diff, ddof=1) * 100
    se_d = std_d / np.sqrt(n)
    t_stat, p_val = stats.ttest_1samp(diff, 0.0)
    t_crit = stats.t.ppf(0.975, df=n-1)
    ci_low = mean_d - t_crit * se_d
    ci_high = mean_d + t_crit * se_d
    
    verdict = "Statistically significant" if p_val < 0.05 else "Indistinguishable from noise"
    print(f"{aug:15s} vs. Version A | Mean Diff: {mean_d:+.2f}% | 95% CI: [{ci_low:+.2f}%, {ci_high:+.2f}%] | t={t_stat:.3f}, p={p_val:.4f} | {verdict}")
