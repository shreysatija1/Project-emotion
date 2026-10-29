"""
Candidate C2: Pretrained Speech Embedding Models.
Evaluates:
- facebook/wav2vec2-base
- facebook/hubert-base-ls960
- microsoft/wavlm-base-plus

Across transformer layers (3, 6, 9, 12) and pooling schemes (mean, mean+std).
Classifiers:
- Logistic Regression (tuned C: 0.01, 0.1, 1.0)
- Linear SVM with calibration
- MLP Classifier with early stopping and dropout

Protocols:
- P1: 5-fold speaker-disjoint cross-validation
- P2: 3-fold Leave-One-Corpus-Out
Seeds: 42, 43, 44.
Selection of best layer, pooling, and classifier strictly on validation speakers!
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier

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
from embeddings_c2 import extract_embeddings_for_dataset

ENCODER_MODELS = [
    "facebook/wav2vec2-base",
    "facebook/hubert-base-ls960",
    "microsoft/wavlm-base-plus"
]


def get_embedding_classifiers(seed=42):
    """Candidate heads for pretrained embeddings."""
    return {
        "logreg_c0.1": LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=seed),
        "logreg_c1.0": LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=seed),
        "linear_svm_c0.1": CalibratedClassifierCV(LinearSVC(C=0.1, dual=False, class_weight="balanced", random_state=seed)),
        "mlp_small": MLPClassifier(hidden_layer_sizes=(128,), early_stopping=True, max_iter=150, random_state=seed)
    }


def run_c2_experiments():
    df = get_unified_metadata()
    y_all = df["target_label"].values
    
    p1_folds = get_p1_folds()
    p2_folds = get_p2_folds()
    seeds = [42, 43, 44]
    
    p1_all_rows = []
    p2_all_rows = []
    
    for model_name in ENCODER_MODELS:
        encoder_tag = model_name.split("/")[-1]
        print(f"\n" + "=" * 60)
        print(f"EVALUATING ENCODER: {encoder_tag} ({model_name})")
        print("=" * 60)
        
        try:
            embeddings_dict = extract_embeddings_for_dataset(model_name, batch_size=16)
        except Exception as e:
            print(f"Warning: Failed to extract embeddings for {model_name}: {e}")
            continue
            
        rep_keys = list(embeddings_dict.keys())
        print(f"Representations available ({len(rep_keys)}): {rep_keys}")
        
        # ---------------------------------------------------------------------
        # PROTOCOL P1 (Speaker-Disjoint CV)
        # ---------------------------------------------------------------------
        print(f"\n--- Running P1 on {encoder_tag} ---")
        for fold in p1_folds:
            f_idx = fold["fold"]
            tr_idx, va_idx, te_idx = get_split_indices_for_p1(df, fold, include_mesd_in_train=False)
            
            y_tr = y_all[tr_idx]
            y_va = y_all[va_idx]
            y_te = y_all[te_idx]
            
            for seed in seeds:
                # Select best (rep_key, classifier) strictly on validation
                best_va_f1 = -1.0
                best_rep_key = None
                best_clf_name = None
                best_va_probs = None
                best_fitted_clf = None
                best_fitted_scaler = None
                
                # Check representations
                for rk in rep_keys:
                    X_curr = embeddings_dict[rk]
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_curr[tr_idx])
                    X_va = scaler.transform(X_curr[va_idx])
                    
                    clfs = get_embedding_classifiers(seed=seed)
                    for c_name, clf in clfs.items():
                        clf.fit(X_tr, y_tr)
                        va_p = clf.predict_proba(X_va)
                        va_res = evaluate_predictions(y_va, va_p)
                        if va_res["macro_f1"] > best_va_f1:
                            best_va_f1 = va_res["macro_f1"]
                            best_rep_key = rk
                            best_clf_name = c_name
                            best_va_probs = va_p
                            best_fitted_clf = clf
                            best_fitted_scaler = scaler
                            
                # Score test exactly once on the frozen chosen configuration
                X_best = embeddings_dict[best_rep_key]
                X_te = best_fitted_scaler.transform(X_best[te_idx])
                te_p = best_fitted_clf.predict_proba(X_te)
                
                va_metrics = evaluate_predictions(y_va, best_va_probs)
                te_metrics = evaluate_predictions(y_te, te_p)
                
                p1_all_rows.append({
                    "candidate": f"C2_{encoder_tag}",
                    "encoder": encoder_tag,
                    "selected_layer": best_rep_key,
                    "selected_classifier": best_clf_name,
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
                print(f"[C2 P1 {encoder_tag} Fold {f_idx} Seed {seed}] Best: {best_rep_key} + {best_clf_name} | Val F1: {va_metrics['macro_f1']:.3f} | Test F1: {te_metrics['macro_f1']:.3f} | Test Acc: {te_metrics['accuracy']:.3f}")
                
        # ---------------------------------------------------------------------
        # PROTOCOL P2 (Leave-One-Corpus-Out)
        # ---------------------------------------------------------------------
        print(f"\n--- Running P2 on {encoder_tag} ---")
        for loco in p2_folds:
            f_name = loco["fold_name"]
            tr_idx, va_idx, te_idx = get_split_indices_for_p2(df, loco, include_mesd_in_train=False)
            
            y_tr = y_all[tr_idx]
            y_va = y_all[va_idx]
            y_te = y_all[te_idx]
            
            for seed in seeds:
                best_va_f1 = -1.0
                best_rep_key = None
                best_clf_name = None
                best_va_probs = None
                best_fitted_clf = None
                best_fitted_scaler = None
                
                for rk in rep_keys:
                    X_curr = embeddings_dict[rk]
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_curr[tr_idx])
                    X_va = scaler.transform(X_curr[va_idx])
                    
                    clfs = get_embedding_classifiers(seed=seed)
                    for c_name, clf in clfs.items():
                        clf.fit(X_tr, y_tr)
                        va_p = clf.predict_proba(X_va)
                        va_res = evaluate_predictions(y_va, va_p)
                        if va_res["macro_f1"] > best_va_f1:
                            best_va_f1 = va_res["macro_f1"]
                            best_rep_key = rk
                            best_clf_name = c_name
                            best_va_probs = va_p
                            best_fitted_clf = clf
                            best_fitted_scaler = scaler
                            
                X_best = embeddings_dict[best_rep_key]
                X_te = best_fitted_scaler.transform(X_best[te_idx])
                te_p = best_fitted_clf.predict_proba(X_te)
                
                va_metrics = evaluate_predictions(y_va, best_va_probs)
                te_metrics = evaluate_predictions(y_te, te_p)
                
                p2_all_rows.append({
                    "candidate": f"C2_{encoder_tag}",
                    "encoder": encoder_tag,
                    "selected_layer": best_rep_key,
                    "selected_classifier": best_clf_name,
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
                print(f"[C2 P2 {encoder_tag} {f_name} Seed {seed}] Best: {best_rep_key} + {best_clf_name} | Val F1: {va_metrics['macro_f1']:.3f} | Test F1: {te_metrics['macro_f1']:.3f} | Test Acc: {te_metrics['accuracy']:.3f}")
                
    p1_df = pd.DataFrame(p1_all_rows)
    p2_df = pd.DataFrame(p2_all_rows)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p1_df.to_csv(RESULTS_DIR / "c2_p1_results.csv", index=False)
    p2_df.to_csv(RESULTS_DIR / "c2_p2_results.csv", index=False)
    
    print("\n" + "=" * 60)
    print("C2 PRETRAINED EMBEDDINGS SUMMARY")
    print("=" * 60)
    for model_name in ENCODER_MODELS:
        enc = model_name.split("/")[-1]
        sub1 = p1_df[p1_df["encoder"] == enc]
        sub2 = p2_df[p2_df["encoder"] == enc]
        if len(sub1) > 0 and len(sub2) > 0:
            print(f"[{enc}] P1 Test F1: {sub1['test_macro_f1'].mean():.3f} +/- {sub1['test_macro_f1'].std():.3f} (Acc: {sub1['test_accuracy'].mean():.3f}) | P2 Test F1: {sub2['test_macro_f1'].mean():.3f} +/- {sub2['test_macro_f1'].std():.3f} (Acc: {sub2['test_accuracy'].mean():.3f})")


if __name__ == "__main__":
    run_c2_experiments()
