"""
Candidate C1: Deployable Hand-crafted Acoustic Models.
Evaluates both eGeMAPSv02 (88-dim) and Comprehensive Librosa stats (442-dim),
and their combination across diverse classifiers:
- Logistic Regression (tuned C)
- Linear SVM with calibration
- RBF SVM with calibration (tuned C, gamma)
- Gradient Boosting (LightGBM with balanced weights)
- Small MLP with early stopping

Protocol:
- Scalers fitted strictly on training data of each fold!
- Model selection and hyperparameter tuning performed strictly on validation speakers!
- Test scored once per fold on the frozen best model.
- Evaluated on P1 (5-fold CV) and P2 (LOCO) across seeds 42, 43, 44.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier

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
from features_opensmile import get_cached_egemaps_features, get_cached_librosa_features


def get_candidate_models(seed=42):
    """Return dictionary of candidate models with their hyperparameter grids."""
    models = {
        "logreg_c0.1": LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=seed),
        "logreg_c1.0": LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=seed),
        "linear_svm_c0.1": CalibratedClassifierCV(LinearSVC(C=0.1, dual=False, class_weight="balanced", random_state=seed)),
        "rbf_svm_c1.0": SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
        "rbf_svm_c5.0": SVC(C=5.0, kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
        "lightgbm": LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, class_weight="balanced", random_state=seed, verbose=-1),
        "mlp_small": MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=150, random_state=seed)
    }
    return models


def run_c1_experiments():
    df = get_unified_metadata()
    y_all = df["target_label"].values
    
    # 1. Load or extract features
    print("\n[C1] Loading / Extracting Handcrafted Feature Sets...")
    X_egemaps = get_cached_egemaps_features(df)
    X_librosa = get_cached_librosa_features(df)
    X_combined = np.hstack([X_egemaps, X_librosa])
    
    feature_sets = {
        "egemaps_88": X_egemaps,
        "librosa_442": X_librosa,
        "combined_530": X_combined
    }
    
    p1_folds = get_p1_folds()
    p2_folds = get_p2_folds()
    seeds = [42, 43, 44]
    
    # =========================================================================
    # PART 1: PROTOCOL P1 (Speaker-Disjoint CV)
    # =========================================================================
    print("\n" + "=" * 60)
    print("RUNNING CANDIDATE C1 ON PROTOCOL P1 (SPEAKER-DISJOINT CV)")
    print("=" * 60)
    
    p1_rows = []
    
    for feat_name, X_feat in feature_sets.items():
        print(f"\n--- Evaluating Feature Set: {feat_name} on P1 ---")
        for fold in p1_folds:
            f_idx = fold["fold"]
            tr_idx, va_idx, te_idx = get_split_indices_for_p1(df, fold, include_mesd_in_train=False)
            
            # Scaler on train only
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_feat[tr_idx])
            X_va = scaler.transform(X_feat[va_idx])
            X_te = scaler.transform(X_feat[te_idx])
            
            y_tr = y_all[tr_idx]
            y_va = y_all[va_idx]
            y_te = y_all[te_idx]
            
            for seed in seeds:
                models = get_candidate_models(seed=seed)
                
                # Pick best model configuration on validation Macro-F1 ONLY
                best_model_name = None
                best_va_f1 = -1.0
                best_va_probs = None
                best_fitted_model = None
                
                for m_name, model in models.items():
                    model.fit(X_tr, y_tr)
                    va_probs = model.predict_proba(X_va)
                    va_eval = evaluate_predictions(y_va, va_probs)
                    if va_eval["macro_f1"] > best_va_f1:
                        best_va_f1 = va_eval["macro_f1"]
                        best_model_name = m_name
                        best_va_probs = va_probs
                        best_fitted_model = model
                        
                # Score test exactly once on the selected best validation model
                te_probs = best_fitted_model.predict_proba(X_te)
                va_metrics = evaluate_predictions(y_va, best_va_probs)
                te_metrics = evaluate_predictions(y_te, te_probs)
                
                p1_rows.append({
                    "candidate": f"C1_{feat_name}",
                    "selected_classifier": best_model_name,
                    "protocol": "P1",
                    "fold": f_idx,
                    "seed": seed,
                    "val_macro_f1": va_metrics["macro_f1"],
                    "val_uar": va_metrics["uar"],
                    "val_accuracy": va_metrics["accuracy"],
                    "val_ece": va_metrics["ece"],
                    "test_macro_f1": te_metrics["macro_f1"],
                    "test_uar": te_metrics["uar"],
                    "test_accuracy": te_metrics["accuracy"],
                    "test_ece": te_metrics["ece"]
                })
                print(f"[C1 P1 {feat_name} Fold {f_idx} Seed {seed}] Best: {best_model_name} | Val F1: {va_metrics['macro_f1']:.3f} | Test F1: {te_metrics['macro_f1']:.3f} | Test Acc: {te_metrics['accuracy']:.3f}")
                
    p1_df = pd.DataFrame(p1_rows)
    p1_df.to_csv(RESULTS_DIR / "c1_p1_results.csv", index=False)
    
    print("\n--- P1 Handcrafted Summary ---")
    for feat_name in feature_sets.keys():
        sub = p1_df[p1_df["candidate"] == f"C1_{feat_name}"]
        print(f"[{feat_name}] Val F1: {sub['val_macro_f1'].mean():.3f} +/- {sub['val_macro_f1'].std():.3f} | Test F1: {sub['test_macro_f1'].mean():.3f} +/- {sub['test_macro_f1'].std():.3f} | Test Acc: {sub['test_accuracy'].mean():.3f}")

    # =========================================================================
    # PART 2: PROTOCOL P2 (Leave-One-Corpus-Out)
    # =========================================================================
    print("\n" + "=" * 60)
    print("RUNNING CANDIDATE C1 ON PROTOCOL P2 (LEAVE-ONE-CORPUS-OUT)")
    print("=" * 60)
    
    p2_rows = []
    
    for feat_name, X_feat in feature_sets.items():
        print(f"\n--- Evaluating Feature Set: {feat_name} on P2 ---")
        for loco in p2_folds:
            f_name = loco["fold_name"]
            tr_idx, va_idx, te_idx = get_split_indices_for_p2(df, loco, include_mesd_in_train=False)
            
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_feat[tr_idx])
            X_va = scaler.transform(X_feat[va_idx])
            X_te = scaler.transform(X_feat[te_idx])
            
            y_tr = y_all[tr_idx]
            y_va = y_all[va_idx]
            y_te = y_all[te_idx]
            
            for seed in seeds:
                models = get_candidate_models(seed=seed)
                
                best_model_name = None
                best_va_f1 = -1.0
                best_va_probs = None
                best_fitted_model = None
                
                for m_name, model in models.items():
                    model.fit(X_tr, y_tr)
                    va_probs = model.predict_proba(X_va)
                    va_eval = evaluate_predictions(y_va, va_probs)
                    if va_eval["macro_f1"] > best_va_f1:
                        best_va_f1 = va_eval["macro_f1"]
                        best_model_name = m_name
                        best_va_probs = va_probs
                        best_fitted_model = model
                        
                te_probs = best_fitted_model.predict_proba(X_te)
                va_metrics = evaluate_predictions(y_va, best_va_probs)
                te_metrics = evaluate_predictions(y_te, te_probs)
                
                p2_rows.append({
                    "candidate": f"C1_{feat_name}",
                    "selected_classifier": best_model_name,
                    "protocol": "P2",
                    "fold": f_name,
                    "seed": seed,
                    "val_macro_f1": va_metrics["macro_f1"],
                    "val_uar": va_metrics["uar"],
                    "val_accuracy": va_metrics["accuracy"],
                    "val_ece": va_metrics["ece"],
                    "test_macro_f1": te_metrics["macro_f1"],
                    "test_uar": te_metrics["uar"],
                    "test_accuracy": te_metrics["accuracy"],
                    "test_ece": te_metrics["ece"]
                })
                print(f"[C1 P2 {feat_name} {f_name} Seed {seed}] Best: {best_model_name} | Val F1: {va_metrics['macro_f1']:.3f} | Test F1: {te_metrics['macro_f1']:.3f} | Test Acc: {te_metrics['accuracy']:.3f}")
                
    p2_df = pd.DataFrame(p2_rows)
    p2_df.to_csv(RESULTS_DIR / "c1_p2_results.csv", index=False)
    
    print("\n--- P2 Handcrafted Summary ---")
    for feat_name in feature_sets.keys():
        sub = p2_df[p2_df["candidate"] == f"C1_{feat_name}"]
        print(f"[{feat_name}] Val F1: {sub['val_macro_f1'].mean():.3f} +/- {sub['val_macro_f1'].std():.3f} | Test F1: {sub['test_macro_f1'].mean():.3f} +/- {sub['test_macro_f1'].std():.3f} | Test Acc: {sub['test_accuracy'].mean():.3f}")


if __name__ == "__main__":
    run_c1_experiments()
