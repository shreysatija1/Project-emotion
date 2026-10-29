"""
Candidate C4: Web-Condition Robustness and Microphone Simulation Study.
Objective:
- Evaluate the vulnerability of trained SER models to realistic web recording conditions:
  gain shift, ambient colored noise (12-28 dB SNR), early room reverberation,
  bandpass frequency response (80 Hz - 7 kHz), and mild saturation.
- Test whether training-time microphone simulation augmentation closes the robustness gap
  without degrading clean speech accuracy.
- Protocols: P1 (Speaker-Disjoint CV) and P2 (Leave-One-Corpus-Out) on eGeMAPSv02 (88 dims).
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import opensmile
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
    get_split_indices_for_p1, get_split_indices_for_p2,
    load_clip, SR
)
from metrics import evaluate_predictions
from mic_simulation import simulate_web_microphone


def get_or_extract_micsim_egemaps():
    """Extract eGeMAPS 88 features on mic-simulated audio waveforms."""
    cache_path = CACHE_DIR / "egemaps_88_micsim.npy"
    if cache_path.exists():
        print(f"[C4] Loading cached mic-simulated eGeMAPS from {cache_path}...")
        return np.load(cache_path)
        
    print("[C4] Extracting eGeMAPS 88 on mic-simulated waveforms...")
    df = get_unified_metadata()
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals
    )
    
    feats = []
    rng = np.random.default_rng(42)
    
    for idx, row in df.iterrows():
        y = load_clip(row["audio_path"], target_sr=SR, max_duration=10.0)
        y_sim = simulate_web_microphone(y, sr=SR, rng=rng)
        
        # openSMILE accepts 1D numpy array with sampling_rate
        f = smile.process_signal(y_sim, SR)
        feats.append(f.values[0])
        
        if (idx + 1) % 500 == 0 or (idx + 1) == len(df):
            print(f"  Processed {idx + 1}/{len(df)} clips...")
            
    feats_arr = np.array(feats, dtype=np.float32)
    np.save(cache_path, feats_arr)
    print(f"[C4] Saved cached mic-simulated eGeMAPS to {cache_path}")
    return feats_arr


def run_c4_study():
    df = get_unified_metadata()
    y_all = df["target_label"].values
    
    # Load clean eGeMAPS and mic-simulated eGeMAPS
    X_clean = np.load(CACHE_DIR / "egemaps_88_features.npy")
    X_micsim = get_or_extract_micsim_egemaps()
    
    p1_folds = get_p1_folds()
    p2_folds = get_p2_folds()
    seeds = [42, 43, 44]
    
    c4_rows = []
    
    print("\n" + "=" * 60)
    print("RUNNING CANDIDATE C4: WEB ROBUSTNESS EVALUATION (P1 & P2)")
    print("=" * 60)
    
    # We evaluate 2 training regimes:
    # 1. 'clean_only': trained on clean train audio
    # 2. 'mic_augmented': trained on clean train + mic-simulated train audio (2x training size)
    # And test on:
    # A. Clean test audio
    # B. Mic-simulated test audio
    
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
            
            # Regime 1: Clean train only
            scaler_clean = StandardScaler()
            X_tr_clean = scaler_clean.fit_transform(X_clean[tr_idx])
            X_va_clean = scaler_clean.transform(X_clean[va_idx])
            X_te_clean = scaler_clean.transform(X_clean[te_idx])
            X_te_mic = scaler_clean.transform(X_micsim[te_idx])
            
            # Regime 2: Mic-augmented train (clean + mic-simulated train clips)
            X_tr_aug = np.vstack([X_clean[tr_idx], X_micsim[tr_idx]])
            y_tr_aug = np.concatenate([y_tr, y_tr])
            scaler_aug = StandardScaler()
            X_tr_aug_scaled = scaler_aug.fit_transform(X_tr_aug)
            X_te_clean_aug = scaler_aug.transform(X_clean[te_idx])
            X_te_mic_aug = scaler_aug.transform(X_micsim[te_idx])
            
            for seed in seeds:
                # 1. Baseline Clean Model
                clf_clean = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=seed)
                clf_clean.fit(X_tr_clean, y_tr)
                
                te_clean_p = clf_clean.predict_proba(X_te_clean)
                te_mic_p = clf_clean.predict_proba(X_te_mic)
                
                m_clean_clean = evaluate_predictions(y_te, te_clean_p)
                m_clean_mic = evaluate_predictions(y_te, te_mic_p)
                
                c4_rows.append({
                    "protocol": proto_name,
                    "fold": f_id,
                    "seed": seed,
                    "train_regime": "clean_only",
                    "test_condition": "clean",
                    "macro_f1": m_clean_clean["macro_f1"],
                    "accuracy": m_clean_clean["accuracy"],
                    "ece": m_clean_clean["ece"]
                })
                c4_rows.append({
                    "protocol": proto_name,
                    "fold": f_id,
                    "seed": seed,
                    "train_regime": "clean_only",
                    "test_condition": "mic_simulated",
                    "macro_f1": m_clean_mic["macro_f1"],
                    "accuracy": m_clean_mic["accuracy"],
                    "ece": m_clean_mic["ece"]
                })
                
                # 2. Mic-Augmented Model
                clf_aug = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=seed)
                clf_aug.fit(X_tr_aug_scaled, y_tr_aug)
                
                te_clean_aug_p = clf_aug.predict_proba(X_te_clean_aug)
                te_mic_aug_p = clf_aug.predict_proba(X_te_mic_aug)
                
                m_aug_clean = evaluate_predictions(y_te, te_clean_aug_p)
                m_aug_mic = evaluate_predictions(y_te, te_mic_aug_p)
                
                c4_rows.append({
                    "protocol": proto_name,
                    "fold": f_id,
                    "seed": seed,
                    "train_regime": "mic_augmented",
                    "test_condition": "clean",
                    "macro_f1": m_aug_clean["macro_f1"],
                    "accuracy": m_aug_clean["accuracy"],
                    "ece": m_aug_clean["ece"]
                })
                c4_rows.append({
                    "protocol": proto_name,
                    "fold": f_id,
                    "seed": seed,
                    "train_regime": "mic_augmented",
                    "test_condition": "mic_simulated",
                    "macro_f1": m_aug_mic["macro_f1"],
                    "accuracy": m_aug_mic["accuracy"],
                    "ece": m_aug_mic["ece"]
                })
                
    c4_df = pd.DataFrame(c4_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    c4_df.to_csv(RESULTS_DIR / "c4_robustness_results.csv", index=False)
    
    print("\n" + "=" * 60)
    print("C4 WEB ROBUSTNESS RESULTS SUMMARY")
    print("=" * 60)
    for proto in ["P1", "P2"]:
        print(f"\n--- Protocol {proto} ---")
        for reg in ["clean_only", "mic_augmented"]:
            for cond in ["clean", "mic_simulated"]:
                sub = c4_df[(c4_df["protocol"] == proto) & (c4_df["train_regime"] == reg) & (c4_df["test_condition"] == cond)]
                print(f"[{reg} -> {cond}] Macro-F1: {sub['macro_f1'].mean():.3f} +/- {sub['macro_f1'].std():.3f} | Acc: {sub['accuracy'].mean():.3f} +/- {sub['accuracy'].std():.3f}")


if __name__ == "__main__":
    run_c4_study()
