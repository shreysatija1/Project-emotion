"""
Candidate C2: Pretrained Speech Embedding Layer & Pooling Ablation Study (wav2vec2-base).
Evaluates:
- Transformer Layers: 3 (early), 6 (mid), 9 (mid-late), 12 (final)
- Pooling Strategies: Mean Pooling (768 dims) vs Mean + Std Pooling (1536 dims)
- Protocols: P1 (5-fold speaker-disjoint CV) and P2 (3-fold Leave-One-Corpus-Out)
- Seeds: 42, 43, 44
Outputs:
- week6_webmodel/results/c2_p1_results.csv
- week6_webmodel/results/c2_p2_results.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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


def run_c2_wav2vec2_ablation():
    df = get_unified_metadata()
    y_all = df["target_label"].values
    
    npz_path = CACHE_DIR / "embeddings_facebook_wav2vec2_base.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Cached wav2vec2 embeddings not found at {npz_path}")
        
    data = np.load(npz_path)
    rep_keys = list(data.files)
    print(f"[C2 Ablation] Loaded wav2vec2-base representations: {rep_keys}")
    
    p1_folds = get_p1_folds()
    p2_folds = get_p2_folds()
    seeds = [42, 43, 44]
    
    p1_rows = []
    p2_rows = []
    
    print("\n" + "=" * 60)
    print("RUNNING C2 WAV2VEC2 LAYER & POOLING ABLATION (P1)")
    print("=" * 60)
    
    for rk in rep_keys:
        layer_num = int(rk.split("_")[1])
        pool_type = "mean_std" if "mean_std" in rk else "mean"
        X_curr = data[rk]
        
        for fold in p1_folds:
            f_idx = fold["fold"]
            tr_idx, va_idx, te_idx = get_split_indices_for_p1(df, fold, include_mesd_in_train=False)
            
            y_tr = y_all[tr_idx]
            y_va = y_all[va_idx]
            y_te = y_all[te_idx]
            
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_curr[tr_idx])
            X_va = scaler.transform(X_curr[va_idx])
            X_te = scaler.transform(X_curr[te_idx])
            
            for seed in seeds:
                clf = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=seed)
                clf.fit(X_tr, y_tr)
                
                va_p = clf.predict_proba(X_va)
                te_p = clf.predict_proba(X_te)
                
                va_res = evaluate_predictions(y_va, va_p)
                te_res = evaluate_predictions(y_te, te_p)
                
                p1_rows.append({
                    "encoder": "wav2vec2-base",
                    "rep_key": rk,
                    "layer": layer_num,
                    "pooling": pool_type,
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
                
    print("\n" + "=" * 60)
    print("RUNNING C2 WAV2VEC2 LAYER & POOLING ABLATION (P2 LOCO)")
    print("=" * 60)
    
    for rk in rep_keys:
        layer_num = int(rk.split("_")[1])
        pool_type = "mean_std" if "mean_std" in rk else "mean"
        X_curr = data[rk]
        
        for loco in p2_folds:
            f_name = loco["fold_name"]
            tr_idx, va_idx, te_idx = get_split_indices_for_p2(df, loco, include_mesd_in_train=False)
            
            y_tr = y_all[tr_idx]
            y_va = y_all[va_idx]
            y_te = y_all[te_idx]
            
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_curr[tr_idx])
            X_va = scaler.transform(X_curr[va_idx])
            X_te = scaler.transform(X_curr[te_idx])
            
            for seed in seeds:
                clf = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=seed)
                clf.fit(X_tr, y_tr)
                
                va_p = clf.predict_proba(X_va)
                te_p = clf.predict_proba(X_te)
                
                va_res = evaluate_predictions(y_va, va_p)
                te_res = evaluate_predictions(y_te, te_p)
                
                p2_rows.append({
                    "encoder": "wav2vec2-base",
                    "rep_key": rk,
                    "layer": layer_num,
                    "pooling": pool_type,
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
                
    p1_df = pd.DataFrame(p1_rows)
    p2_df = pd.DataFrame(p2_rows)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p1_df.to_csv(RESULTS_DIR / "c2_p1_results.csv", index=False)
    p2_df.to_csv(RESULTS_DIR / "c2_p2_results.csv", index=False)
    
    print("\n" + "=" * 60)
    print("C2 WAV2VEC2 LAYER & POOLING ABLATION SUMMARY")
    print("=" * 60)
    print("\n--- Protocol P1 (Speaker-Disjoint CV) ---")
    for rk in rep_keys:
        sub = p1_df[p1_df["rep_key"] == rk]
        print(f"[{rk}] Test Macro-F1: {sub['test_macro_f1'].mean():.3f} +/- {sub['test_macro_f1'].std():.3f} | Test Acc: {sub['test_accuracy'].mean():.3f} +/- {sub['test_accuracy'].std():.3f}")
        
    print("\n--- Protocol P2 (Leave-One-Corpus-Out) ---")
    for rk in rep_keys:
        sub = p2_df[p2_df["rep_key"] == rk]
        print(f"[{rk}] Test Macro-F1: {sub['test_macro_f1'].mean():.3f} +/- {sub['test_macro_f1'].std():.3f} | Test Acc: {sub['test_accuracy'].mean():.3f} +/- {sub['test_accuracy'].std():.3f}")


if __name__ == "__main__":
    run_c2_wav2vec2_ablation()
