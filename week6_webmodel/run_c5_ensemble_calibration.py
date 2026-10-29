"""
Candidate C5: Ensemble, Temperature Calibration, and Selective Rejection.
Objective:
- Ensemble top handcrafted (C1) and pretrained (C2) models via soft voting.
- Post-hoc probability calibration: Temperature scaling fitted on validation speakers ONLY.
- Evaluate Expected Calibration Error (ECE) and Brier score before vs after calibration on frozen test sets.
- Selective prediction / rejection: Evaluate accuracy and Macro-F1 at 0%, 10%, 20%, 30% rejection rates
  based on prediction entropy H(p) = -sum(p * log(p)).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
from metrics import evaluate_predictions, compute_ece


class TemperatureCalibrator:
    """Post-hoc temperature scaling fitted strictly on validation logits."""
    def __init__(self):
        self.temperature = 1.0
        
    def fit(self, val_logits, y_val):
        """Optimize temperature T to minimize negative log likelihood on validation set."""
        def nll(t):
            t = t[0]
            scaled = val_logits / max(t, 1e-3)
            # Softmax
            exp_s = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
            probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
            # Cross entropy
            eps = 1e-12
            loss = -np.mean(np.log(probs[np.arange(len(y_val)), y_val] + eps))
            return loss
            
        res = minimize(nll, [1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        return self
        
    def calibrate(self, logits):
        scaled = logits / max(self.temperature, 1e-3)
        exp_s = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_s / np.sum(exp_s, axis=1, keepdims=True)


def compute_rejection_curve(y_true, probs, reject_rates=[0.0, 0.1, 0.2, 0.3]):
    """
    Evaluate accuracy and macro-F1 when rejecting the clips with highest entropy.
    """
    eps = 1e-12
    entropies = -np.sum(probs * np.log(probs + eps), axis=1)
    
    curve = []
    n = len(y_true)
    for rate in reject_rates:
        k = int(np.floor(n * (1.0 - rate)))
        # Keep clips with lowest entropy (highest certainty)
        keep_indices = np.argsort(entropies)[:k]
        
        y_sub = y_true[keep_indices]
        p_sub = probs[keep_indices]
        m = evaluate_predictions(y_sub, p_sub)
        
        curve.append({
            "reject_rate": rate,
            "coverage": 1.0 - rate,
            "n_samples": k,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "uar": m["uar"]
        })
    return curve


def run_c5_study():
    df = get_unified_metadata()
    y_all = df["target_label"].values
    
    # Load features
    c1_egemaps = np.load(CACHE_DIR / "egemaps_88_features.npy")
    c1_librosa = np.load(CACHE_DIR / "librosa_442_features.npy")
    X_c1 = np.hstack([c1_egemaps, c1_librosa])  # Combined 530
    
    # Check if wav2vec2 embeddings exist
    w2v2_path = CACHE_DIR / "embeddings_facebook_wav2vec2_base.npz"
    if w2v2_path.exists():
        data = np.load(w2v2_path)
        # Use layer 9 or 12 mean_std
        rk = "layer_9_mean_std" if "layer_9_mean_std" in data else list(data.files)[0]
        X_c2 = data[rk]
    else:
        print("[C5] Pretrained embeddings not found, using eGeMAPS 88 as second model.")
        X_c2 = c1_egemaps
        
    p1_folds = get_p1_folds()
    p2_folds = get_p2_folds()
    seeds = [42, 43, 44]
    
    ensemble_rows = []
    rejection_rows = []
    
    print("\n" + "=" * 60)
    print("RUNNING CANDIDATE C5: ENSEMBLE, CALIBRATION & REJECTION")
    print("=" * 60)
    
    for proto_name, folds, get_split_fn in [
        ("P1", p1_folds, get_split_indices_for_p1),
        ("P2", p2_folds, get_split_indices_for_p2)
    ]:
        print(f"\n--- Protocol {proto_name} ---")
        for fold in folds:
            f_id = fold["fold"] if "fold" in fold else fold["fold_name"]
            tr_idx, va_idx, te_idx = get_split_fn(df, fold, include_mesd_in_train=False)
            
            y_tr = y_all[tr_idx]
            y_va = y_all[va_idx]
            y_te = y_all[te_idx]
            
            # Scalers fit on train only
            scaler1 = StandardScaler()
            X_tr1 = scaler1.fit_transform(X_c1[tr_idx])
            X_va1 = scaler1.transform(X_c1[va_idx])
            X_te1 = scaler1.transform(X_c1[te_idx])
            
            scaler2 = StandardScaler()
            X_tr2 = scaler2.fit_transform(X_c2[tr_idx])
            X_va2 = scaler2.transform(X_c2[va_idx])
            X_te2 = scaler2.transform(X_c2[te_idx])
            
            for seed in seeds:
                # Model 1: C1 MLP / LogReg
                clf1 = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=seed)
                clf1.fit(X_tr1, y_tr)
                va_p1 = clf1.predict_proba(X_va1)
                te_p1 = clf1.predict_proba(X_te1)
                
                # Model 2: C2 LogReg
                clf2 = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=seed)
                clf2.fit(X_tr2, y_tr)
                va_p2 = clf2.predict_proba(X_va2)
                te_p2 = clf2.predict_proba(X_te2)
                
                # Single model 1 metrics
                m1_te = evaluate_predictions(y_te, te_p1)
                
                # Single model 2 metrics
                m2_te = evaluate_predictions(y_te, te_p2)
                
                # Ensemble: Soft voting (average probabilities)
                va_ens_p = 0.5 * (va_p1 + va_p2)
                te_ens_p = 0.5 * (te_p1 + te_p2)
                m_ens_uncal = evaluate_predictions(y_te, te_ens_p)
                
                # Temperature Scaling: Fit strictly on validation set!
                eps = 1e-12
                va_logits = np.log(va_ens_p + eps)
                te_logits = np.log(te_ens_p + eps)
                
                calibrator = TemperatureCalibrator().fit(va_logits, y_va)
                te_cal_p = calibrator.calibrate(te_logits)
                m_ens_cal = evaluate_predictions(y_te, te_cal_p)
                
                ensemble_rows.append({
                    "protocol": proto_name,
                    "fold": f_id,
                    "seed": seed,
                    "model_1_te_f1": m1_te["macro_f1"],
                    "model_1_te_acc": m1_te["accuracy"],
                    "model_2_te_f1": m2_te["macro_f1"],
                    "model_2_te_acc": m2_te["accuracy"],
                    "ensemble_uncal_f1": m_ens_uncal["macro_f1"],
                    "ensemble_uncal_acc": m_ens_uncal["accuracy"],
                    "ensemble_uncal_ece": m_ens_uncal["ece"],
                    "calibrator_temperature": calibrator.temperature,
                    "ensemble_cal_f1": m_ens_cal["macro_f1"],
                    "ensemble_cal_acc": m_ens_cal["accuracy"],
                    "ensemble_cal_ece": m_ens_cal["ece"]
                })
                
                # Rejection curve on calibrated probabilities
                curve = compute_rejection_curve(y_te, te_cal_p)
                for pt in curve:
                    pt.update({"protocol": proto_name, "fold": f_id, "seed": seed})
                    rejection_rows.append(pt)
                    
    ens_df = pd.DataFrame(ensemble_rows)
    rej_df = pd.DataFrame(rejection_rows)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ens_df.to_csv(RESULTS_DIR / "c5_ensemble_calibration_results.csv", index=False)
    rej_df.to_csv(RESULTS_DIR / "c5_rejection_curve.csv", index=False)
    
    print("\n" + "=" * 60)
    print("C5 ENSEMBLE & CALIBRATION SUMMARY")
    print("=" * 60)
    for proto in ["P1", "P2"]:
        sub = ens_df[ens_df["protocol"] == proto]
        print(f"\n--- Protocol {proto} ---")
        print(f"Model 1 (C1) Test F1:          {sub['model_1_te_f1'].mean():.3f} +/- {sub['model_1_te_f1'].std():.3f} (Acc: {sub['model_1_te_acc'].mean():.3f})")
        print(f"Model 2 (C2) Test F1:          {sub['model_2_te_f1'].mean():.3f} +/- {sub['model_2_te_f1'].std():.3f} (Acc: {sub['model_2_te_acc'].mean():.3f})")
        print(f"Ensemble Uncalibrated Test F1: {sub['ensemble_uncal_f1'].mean():.3f} +/- {sub['ensemble_uncal_f1'].std():.3f} (Acc: {sub['ensemble_uncal_acc'].mean():.3f}, ECE: {sub['ensemble_uncal_ece'].mean():.3f})")
        print(f"Ensemble Calibrated Test F1:   {sub['ensemble_cal_f1'].mean():.3f} +/- {sub['ensemble_cal_f1'].std():.3f} (Acc: {sub['ensemble_cal_acc'].mean():.3f}, ECE: {sub['ensemble_cal_ece'].mean():.3f})")
        print(f"Mean Fitted Temperature T:     {sub['calibrator_temperature'].mean():.3f}")
        
    print("\n--- Selective Rejection Performance (P1 Calibrated Ensemble) ---")
    p1_rej = rej_df[rej_df["protocol"] == "P1"]
    for rr in [0.0, 0.1, 0.2, 0.3]:
        sub_rr = p1_rej[p1_rej["reject_rate"] == rr]
        print(f"Reject Rate {int(rr*100)}% (Coverage {int((1-rr)*100)}%): Accuracy = {sub_rr['accuracy'].mean():.3f} +/- {sub_rr['accuracy'].std():.3f} | Macro-F1 = {sub_rr['macro_f1'].mean():.3f} +/- {sub_rr['macro_f1'].std():.3f}")


if __name__ == "__main__":
    run_c5_study()
