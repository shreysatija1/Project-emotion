"""
Candidate C3: Multi-Corpus and MESD Training Study.
Objective:
- Evaluate whether adding MESD (862 clips from Mexican Spanish corpus, without individual speaker IDs)
  to the training split improves or degrades cross-speaker (P1) and cross-corpus (P2) generalization.
- Hard Rule: MESD clips are strictly added to training splits ONLY.
  Validation and test splits never contain MESD.
- Model: Evaluated on top C1 handcrafted features (eGeMAPS 88, Librosa 442, Combined 530)
  across seeds 42, 43, 44.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parents[1]
WEEK6_DIR = ROOT_DIR / "week6_webmodel"
RESULTS_DIR = WEEK6_DIR / "results"
CACHE_DIR = WEEK6_DIR / "cache"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(WEEK6_DIR))

from data_manager import (
    get_unified_metadata, get_p1_folds, get_p2_folds,
    get_split_indices_for_p1, get_split_indices_for_p2
)
from metrics import evaluate_predictions


def run_c3_study():
    df = get_unified_metadata()
    y_all = df["target_label"].values
    
    # Load cached features
    egemaps = np.load(CACHE_DIR / "egemaps_88_features.npy")
    librosa_feats = np.load(CACHE_DIR / "librosa_442_features.npy")
    combined_530 = np.hstack([egemaps, librosa_feats])
    
    feature_sets = {
        "C1_egemaps_88": egemaps,
        "C1_combined_530": combined_530
    }
    
    p1_folds = get_p1_folds()
    p2_folds = get_p2_folds()
    seeds = [42, 43, 44]
    
    classifiers = {
        "logreg_c0.1": lambda s: LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=s),
        "mlp_small": lambda s: MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=200, random_state=s)
    }
    
    p1_results = []
    p2_results = []
    
    print("\n" + "=" * 60)
    print("RUNNING CANDIDATE C3: MESD IMPACT STUDY (P1)")
    print("=" * 60)
    
    # ---------------------------------------------------------
    # Protocol P1: Speaker-Disjoint Cross-Validation
    # ---------------------------------------------------------
    for feat_name, X_curr in feature_sets.items():
        for clf_name, clf_fn in classifiers.items():
            for use_mesd in [False, True]:
                mesd_tag = "with_mesd" if use_mesd else "no_mesd"
                print(f"\n[P1] Config: {feat_name} + {clf_name} | MESD in train: {use_mesd}")
                
                for fold in p1_folds:
                    f_idx = fold["fold"]
                    tr_idx, va_idx, te_idx = get_split_indices_for_p1(df, fold, include_mesd_in_train=use_mesd)
                    
                    y_tr = y_all[tr_idx]
                    y_va = y_all[va_idx]
                    y_te = y_all[te_idx]
                    
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_curr[tr_idx])
                    X_va = scaler.transform(X_curr[va_idx])
                    X_te = scaler.transform(X_curr[te_idx])
                    
                    for seed in seeds:
                        model = clf_fn(seed)
                        model.fit(X_tr, y_tr)
                        
                        va_probs = model.predict_proba(X_va)
                        te_probs = model.predict_proba(X_te)
                        
                        va_res = evaluate_predictions(y_va, va_probs)
                        te_res = evaluate_predictions(y_te, te_probs)
                        
                        p1_results.append({
                            "feature_set": feat_name,
                            "classifier": clf_name,
                            "include_mesd_train": use_mesd,
                            "train_n": len(tr_idx),
                            "fold": f_idx,
                            "seed": seed,
                            "val_macro_f1": va_res["macro_f1"],
                            "val_uar": va_res["uar"],
                            "val_accuracy": va_res["accuracy"],
                            "val_ece": va_res["ece"],
                            "test_macro_f1": te_res["macro_f1"],
                            "test_uar": te_res["uar"],
                            "test_accuracy": te_res["accuracy"],
                            "test_ece": te_res["ece"]
                        })
                        
    # ---------------------------------------------------------
    # Protocol P2: Leave-One-Corpus-Out
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("RUNNING CANDIDATE C3: MESD IMPACT STUDY (P2 LOCO)")
    print("=" * 60)
    
    for feat_name, X_curr in feature_sets.items():
        for clf_name, clf_fn in classifiers.items():
            for use_mesd in [False, True]:
                mesd_tag = "with_mesd" if use_mesd else "no_mesd"
                print(f"\n[P2] Config: {feat_name} + {clf_name} | MESD in train: {use_mesd}")
                
                for loco in p2_folds:
                    f_name = loco["fold_name"]
                    tr_idx, va_idx, te_idx = get_split_indices_for_p2(df, loco, include_mesd_in_train=use_mesd)
                    
                    y_tr = y_all[tr_idx]
                    y_va = y_all[va_idx]
                    y_te = y_all[te_idx]
                    
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_curr[tr_idx])
                    X_va = scaler.transform(X_curr[va_idx])
                    X_te = scaler.transform(X_curr[te_idx])
                    
                    for seed in seeds:
                        model = clf_fn(seed)
                        model.fit(X_tr, y_tr)
                        
                        va_probs = model.predict_proba(X_va)
                        te_probs = model.predict_proba(X_te)
                        
                        va_res = evaluate_predictions(y_va, va_probs)
                        te_res = evaluate_predictions(y_te, te_probs)
                        
                        p2_results.append({
                            "feature_set": feat_name,
                            "classifier": clf_name,
                            "include_mesd_train": use_mesd,
                            "train_n": len(tr_idx),
                            "fold": f_name,
                            "seed": seed,
                            "val_macro_f1": va_res["macro_f1"],
                            "val_uar": va_res["uar"],
                            "val_accuracy": va_res["accuracy"],
                            "val_ece": va_res["ece"],
                            "test_macro_f1": te_res["macro_f1"],
                            "test_uar": te_res["uar"],
                            "test_accuracy": te_res["accuracy"],
                            "test_ece": te_res["ece"]
                        })
                        
    p1_df = pd.DataFrame(p1_results)
    p2_df = pd.DataFrame(p2_results)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p1_df.to_csv(RESULTS_DIR / "c3_p1_results.csv", index=False)
    p2_df.to_csv(RESULTS_DIR / "c3_p2_results.csv", index=False)
    
    # Statistical comparison
    print("\n" + "=" * 60)
    print("C3 MESD IMPACT STATISTICAL SUMMARY")
    print("=" * 60)
    
    for proto_name, res_df in [("P1 (Speaker-Disjoint)", p1_df), ("P2 (Leave-One-Corpus-Out)", p2_df)]:
        print(f"\n--- Protocol: {proto_name} ---")
        for f_set in ["C1_egemaps_88", "C1_combined_530"]:
            for c_name in ["logreg_c0.1", "mlp_small"]:
                sub = res_df[(res_df["feature_set"] == f_set) & (res_df["classifier"] == c_name)]
                no_mesd = sub[sub["include_mesd_train"] == False]["test_macro_f1"].values
                with_mesd = sub[sub["include_mesd_train"] == True]["test_macro_f1"].values
                
                diff = with_mesd - no_mesd
                mean_diff = np.mean(diff)
                t_stat, p_val = stats.ttest_rel(with_mesd, no_mesd) if len(diff) > 1 else (0.0, 1.0)
                
                print(f"[{f_set} + {c_name}]:")
                print(f"  No MESD Test F1:   {np.mean(no_mesd):.3f} +/- {np.std(no_mesd):.3f}")
                print(f"  With MESD Test F1: {np.mean(with_mesd):.3f} +/- {np.std(with_mesd):.3f}")
                print(f"  Delta (with - no): {mean_diff:+.3f} (p = {p_val:.4f})")


if __name__ == "__main__":
    run_c3_study()
