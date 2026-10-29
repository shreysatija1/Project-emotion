import pandas as pd
from pathlib import Path

res_dir = Path("week6_webmodel/results")

print("=" * 60)
print("WEEK 6 EMPIRICAL RESULTS SUMMARY ACROSS ALL PROTOCOLS")
print("=" * 60)

print("\n--- B0 Baseline 1D-CNN (Week 3-5 Model on Pooled 6-Class Setup) ---")
b0_p1 = pd.read_csv(res_dir / "b0_p1_results.csv")
b0_p2 = pd.read_csv(res_dir / "b0_p2_results.csv")
print(f"P1 (Speaker-Disjoint):     Test F1 = {b0_p1['test_macro_f1'].mean():.3f} +/- {b0_p1['test_macro_f1'].std():.3f} | Acc = {b0_p1['test_accuracy'].mean():.3f} +/- {b0_p1['test_accuracy'].std():.3f}")
print(f"P2 (Leave-One-Corpus-Out): Test F1 = {b0_p2['test_macro_f1'].mean():.3f} +/- {b0_p2['test_macro_f1'].std():.3f} | Acc = {b0_p2['test_accuracy'].mean():.3f} +/- {b0_p2['test_accuracy'].std():.3f}")

print("\n--- C1 Handcrafted Acoustic Feature Descriptors ---")
c1_p1 = pd.read_csv(res_dir / "c1_p1_results.csv")
c1_p2 = pd.read_csv(res_dir / "c1_p2_results.csv")
for cand in ["C1_egemaps_88", "C1_librosa_442", "C1_combined_530"]:
    s1 = c1_p1[c1_p1["candidate"] == cand]
    s2 = c1_p2[c1_p2["candidate"] == cand]
    print(f"[{cand}]")
    print(f"  P1 (Speaker-Disjoint):     Test F1 = {s1['test_macro_f1'].mean():.3f} +/- {s1['test_macro_f1'].std():.3f} | Acc = {s1['test_accuracy'].mean():.3f} +/- {s1['test_accuracy'].std():.3f}")
    print(f"  P2 (Leave-One-Corpus-Out): Test F1 = {s2['test_macro_f1'].mean():.3f} +/- {s2['test_macro_f1'].std():.3f} | Acc = {s2['test_accuracy'].mean():.3f} +/- {s2['test_accuracy'].std():.3f}")

print("\n--- C2 Pretrained Speech Embeddings (wav2vec2-base) ---")
c2_p1 = pd.read_csv(res_dir / "c2_p1_results.csv")
c2_p2 = pd.read_csv(res_dir / "c2_p2_results.csv")
for rk in ["layer_3_mean_std", "layer_6_mean_std", "layer_9_mean_std", "layer_12_mean_std"]:
    s1 = c2_p1[c2_p1["rep_key"] == rk]
    s2 = c2_p2[c2_p2["rep_key"] == rk]
    print(f"[{rk}]")
    print(f"  P1 (Speaker-Disjoint):     Test F1 = {s1['test_macro_f1'].mean():.3f} +/- {s1['test_macro_f1'].std():.3f} | Acc = {s1['test_accuracy'].mean():.3f} +/- {s1['test_accuracy'].std():.3f}")
    print(f"  P2 (Leave-One-Corpus-Out): Test F1 = {s2['test_macro_f1'].mean():.3f} +/- {s2['test_macro_f1'].std():.3f} | Acc = {s2['test_accuracy'].mean():.3f} +/- {s2['test_accuracy'].std():.3f}")

print("\n--- C3 Multi-Corpus MESD Impact ---")
c3_p1 = pd.read_csv(res_dir / "c3_p1_results.csv")
c3_p2 = pd.read_csv(res_dir / "c3_p2_results.csv")
sub_c3_p1_no = c3_p1[(c3_p1["feature_set"] == "C1_combined_530") & (c3_p1["classifier"] == "mlp_small") & (c3_p1["include_mesd_train"] == False)]
sub_c3_p1_yes = c3_p1[(c3_p1["feature_set"] == "C1_combined_530") & (c3_p1["classifier"] == "mlp_small") & (c3_p1["include_mesd_train"] == True)]
print(f"P1 Combined 530 MLP: Without MESD F1 = {sub_c3_p1_no['test_macro_f1'].mean():.3f} vs With MESD F1 = {sub_c3_p1_yes['test_macro_f1'].mean():.3f} (Delta = {sub_c3_p1_yes['test_macro_f1'].mean() - sub_c3_p1_no['test_macro_f1'].mean():+.3f})")

print("\n--- C4 Web Microphone Simulation Robustness ---")
c4 = pd.read_csv(res_dir / "c4_robustness_results.csv")
for proto in ["P1", "P2"]:
    print(f"Protocol {proto}:")
    for tr in ["clean_only", "mic_augmented"]:
        for te in ["clean", "mic_simulated"]:
            sub = c4[(c4["protocol"] == proto) & (c4["train_regime"] == tr) & (c4["test_condition"] == te)]
            print(f"  {tr} -> {te}: Macro-F1 = {sub['macro_f1'].mean():.3f} +/- {sub['macro_f1'].std():.3f} | Acc = {sub['accuracy'].mean():.3f} +/- {sub['accuracy'].std():.3f}")

print("\n--- C5 Calibrated Ensemble & Rejection Performance ---")
c5 = pd.read_csv(res_dir / "c5_ensemble_calibration_results.csv")
rej = pd.read_csv(res_dir / "c5_rejection_curve.csv")
for proto in ["P1", "P2"]:
    sub = c5[c5["protocol"] == proto]
    print(f"Protocol {proto} Ensemble:")
    print(f"  Uncalibrated: Test F1 = {sub['ensemble_uncal_f1'].mean():.3f} +/- {sub['ensemble_uncal_f1'].std():.3f} | Acc = {sub['ensemble_uncal_acc'].mean():.3f} | ECE = {sub['ensemble_uncal_ece'].mean():.3f}")
    print(f"  Calibrated:   Test F1 = {sub['ensemble_cal_f1'].mean():.3f} +/- {sub['ensemble_cal_f1'].std():.3f} | Acc = {sub['ensemble_cal_acc'].mean():.3f} | ECE = {sub['ensemble_cal_ece'].mean():.3f}")

print("\nP1 Selective Rejection Tradeoff (Calibrated Ensemble):")
p1_rej = rej[rej["protocol"] == "P1"]
for rr in [0.0, 0.1, 0.2, 0.3]:
    sub_rr = p1_rej[p1_rej["reject_rate"] == rr]
    print(f"  Reject {int(rr*100)}% (Coverage {int((1-rr)*100)}%): Accuracy = {sub_rr['accuracy'].mean():.3f} +/- {sub_rr['accuracy'].std():.3f} | Macro-F1 = {sub_rr['macro_f1'].mean():.3f} +/- {sub_rr['macro_f1'].std():.3f}")
