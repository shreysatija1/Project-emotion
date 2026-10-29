"""
Candidate B0: Baseline 1D-CNN reproduction model on the pooled 6-class benchmark.
- Features: MFCC 30 coefficients, pooled mean (30) + std (30) = 60 values (as in Weeks 3-5).
- Model: 1D-CNN (4 Conv1D blocks with BN, ReLU, MaxPool; Dense 120, Dense 6 Softmax).
- Evaluated on P1 (5-fold speaker-disjoint cross-validation) and P2 (Leave-One-Corpus-Out).
- Evaluated across seeds 42, 43, 44.
- StandardScaler fitted on train split only!
- Class weights applied to address the 2.46:1 class imbalance.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

ROOT_DIR = Path(__file__).resolve().parents[1]
WEEK6_DIR = ROOT_DIR / "week6_webmodel"
RESULTS_DIR = WEEK6_DIR / "results"
CACHE_DIR = WEEK6_DIR / "cache"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(WEEK6_DIR))

# Suppress verbose TF logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import keras
from keras import layers

from data_manager import (
    get_unified_metadata, load_clip, get_p1_folds, get_p2_folds,
    get_split_indices_for_p1, get_split_indices_for_p2, SR
)
from metrics import evaluate_predictions


def extract_mfcc_60(y, sr=SR, n_mfcc=30):
    """MFCC pooled over time: mean (30) + std (30) = 60 values."""
    if len(y) == 0:
        return np.zeros(n_mfcc * 2, dtype=np.float32)
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=512, hop_length=256)
    return np.concatenate([m.mean(axis=1), m.std(axis=1)]).astype(np.float32)


def get_cached_b0_features(df):
    """Extract or load cached 60-dim MFCC features for all clips in df."""
    cache_file = CACHE_DIR / "mfcc_60_baseline.npy"
    if cache_file.exists():
        return np.load(cache_file)
        
    print(f"[B0] Extracting 60-dim MFCC features for {len(df)} clips...")
    features = []
    for idx, row in df.iterrows():
        y = load_clip(row["audio_path"], target_sr=SR)
        feat = extract_mfcc_60(y, sr=SR, n_mfcc=30)
        features.append(feat)
    X = np.vstack(features).astype(np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, X)
    print(f"[B0] Saved cached features to {cache_file} (shape: {X.shape})")
    return X


def build_b0_model(num_classes=6, seed=42):
    """Build compiled 1D-CNN baseline model."""
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([
        keras.Input(shape=(60, 1)),
        layers.Conv1D(8, 15, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(2),
        
        layers.Conv1D(16, 15, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(2),
        
        layers.Conv1D(32, 15, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(2),
        
        layers.Conv1D(64, 15, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        
        layers.Flatten(),
        layers.Dense(120, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def run_baseline_p1(df, X_all, seeds=[42, 43, 44]):
    """Run B0 on P1 (5-fold speaker-disjoint cross-validation)."""
    p1_folds = get_p1_folds()
    y_all = df["target_label"].values
    
    rows = []
    print("\n" + "=" * 60)
    print("RUNNING CANDIDATE B0 ON PROTOCOL P1 (SPEAKER-DISJOINT CV)")
    print("=" * 60)
    
    for fold in p1_folds:
        f_idx = fold["fold"]
        tr_idx, va_idx, te_idx = get_split_indices_for_p1(df, fold, include_mesd_in_train=False)
        
        # Fit scaler on train only
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_all[tr_idx])
        X_va = scaler.transform(X_all[va_idx])
        X_te = scaler.transform(X_all[te_idx])
        
        y_tr = y_all[tr_idx]
        y_va = y_all[va_idx]
        y_te = y_all[te_idx]
        
        # Compute class weights on train
        classes = np.unique(y_tr)
        cw = compute_class_weight("balanced", classes=classes, y=y_tr)
        class_weight_dict = {c: float(w) for c, w in zip(classes, cw)}
        
        X_tr_3d = X_tr[..., np.newaxis]
        X_va_3d = X_va[..., np.newaxis]
        X_te_3d = X_te[..., np.newaxis]
        
        for seed in seeds:
            model = build_b0_model(num_classes=6, seed=seed)
            callbacks = [
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
            ]
            model.fit(
                X_tr_3d, y_tr,
                validation_data=(X_va_3d, y_va),
                epochs=40,
                batch_size=32,
                class_weight=class_weight_dict,
                callbacks=callbacks,
                verbose=0
            )
            
            va_probs = model.predict(X_va_3d, verbose=0)
            te_probs = model.predict(X_te_3d, verbose=0)
            
            va_metrics = evaluate_predictions(y_va, va_probs)
            te_metrics = evaluate_predictions(y_te, te_probs)
            
            rows.append({
                "candidate": "B0_1DCNN",
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
            print(f"[B0 P1 Fold {f_idx} Seed {seed}] Val F1: {va_metrics['macro_f1']:.3f}, Test F1: {te_metrics['macro_f1']:.3f}, Test Acc: {te_metrics['accuracy']:.3f}")
            
    res_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(RESULTS_DIR / "b0_p1_results.csv", index=False)
    print(f"\n[B0 P1 Summary] Mean Val F1: {res_df['val_macro_f1'].mean():.3f} +/- {res_df['val_macro_f1'].std():.3f} | Mean Test F1: {res_df['test_macro_f1'].mean():.3f} +/- {res_df['test_macro_f1'].std():.3f} | Test Acc: {res_df['test_accuracy'].mean():.3f}")
    return res_df


def run_baseline_p2(df, X_all, seeds=[42, 43, 44]):
    """Run B0 on P2 (Leave-One-Corpus-Out)."""
    p2_folds = get_p2_folds()
    y_all = df["target_label"].values
    
    rows = []
    print("\n" + "=" * 60)
    print("RUNNING CANDIDATE B0 ON PROTOCOL P2 (LEAVE-ONE-CORPUS-OUT)")
    print("=" * 60)
    
    for loco in p2_folds:
        f_name = loco["fold_name"]
        tr_idx, va_idx, te_idx = get_split_indices_for_p2(df, loco, include_mesd_in_train=False)
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_all[tr_idx])
        X_va = scaler.transform(X_all[va_idx])
        X_te = scaler.transform(X_all[te_idx])
        
        y_tr = y_all[tr_idx]
        y_va = y_all[va_idx]
        y_te = y_all[te_idx]
        
        classes = np.unique(y_tr)
        cw = compute_class_weight("balanced", classes=classes, y=y_tr)
        class_weight_dict = {c: float(w) for c, w in zip(classes, cw)}
        
        X_tr_3d = X_tr[..., np.newaxis]
        X_va_3d = X_va[..., np.newaxis]
        X_te_3d = X_te[..., np.newaxis]
        
        for seed in seeds:
            model = build_b0_model(num_classes=6, seed=seed)
            callbacks = [
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
            ]
            model.fit(
                X_tr_3d, y_tr,
                validation_data=(X_va_3d, y_va),
                epochs=40,
                batch_size=32,
                class_weight=class_weight_dict,
                callbacks=callbacks,
                verbose=0
            )
            
            va_probs = model.predict(X_va_3d, verbose=0)
            te_probs = model.predict(X_te_3d, verbose=0)
            
            va_metrics = evaluate_predictions(y_va, va_probs)
            te_metrics = evaluate_predictions(y_te, te_probs)
            
            rows.append({
                "candidate": "B0_1DCNN",
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
            print(f"[B0 P2 {f_name} Seed {seed}] Val F1: {va_metrics['macro_f1']:.3f}, Test F1: {te_metrics['macro_f1']:.3f}, Test Acc: {te_metrics['accuracy']:.3f}")
            
    res_df = pd.DataFrame(rows)
    res_df.to_csv(RESULTS_DIR / "b0_p2_results.csv", index=False)
    print(f"\n[B0 P2 Summary] Mean Val F1: {res_df['val_macro_f1'].mean():.3f} +/- {res_df['val_macro_f1'].std():.3f} | Mean Test F1: {res_df['test_macro_f1'].mean():.3f} +/- {res_df['test_macro_f1'].std():.3f} | Test Acc: {res_df['test_accuracy'].mean():.3f}")
    return res_df


if __name__ == "__main__":
    df = get_unified_metadata()
    X = get_cached_b0_features(df)
    p1_res = run_baseline_p1(df, X)
    p2_res = run_baseline_p2(df, X)
