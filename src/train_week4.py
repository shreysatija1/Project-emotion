"""
Week 4 Unified Training & Evaluation Engine.

Strict Reproduction Rules:
1. No data leakage: StandardScaler fitted ONLY on training split.
2. Version A: Augments ONLY training clips. Validation and test sets remain clean originals.
3. Model selection based strictly on validation accuracy (ModelCheckpoint).
4. Test set evaluated exactly ONCE on the frozen best validation checkpoint.
5. All runs logged with exact parameters, random seeds, and metrics to results/week4_results.csv.
"""

import os
import sys
import time
import random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
FEAT_DIR = ROOT_DIR / "features"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models" / "week4"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import get_metadata, load_audio_clip, SR
from split_utils import make_stratified_split, make_speaker_independent_split
from features_144 import (
    extract_mfcc_frames144, extract_mfmc_frames144,
    extract_mfcc_stats144, extract_mfmc_stats144
)
from extract_all_features import extract_baseline_mfcc, extract_baseline_mfmc
from model import build_1d_cnn


def set_seed(seed):
    """Set all random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def augment_waveform(y, rng):
    """
    Generate 5 sets of waveforms matching the paper's specification:
    - Set 1: Original clean waveform
    - Set 2: Exact zero-noise copy
    - Set 3: Clean + 0.01 * N(0, 1)
    - Set 4: Clean + 0.025 * N(0, 1)
    - Set 5: Clean + 0.05 * N(0, 1)
    """
    noise_std = [0.0, 0.0, 0.01, 0.025, 0.05]
    copies = []
    for sigma in noise_std:
        if sigma == 0.0:
            copies.append(y.copy())
        else:
            noise = rng.normal(0.0, sigma, size=y.shape).astype(np.float32)
            copies.append(y + noise)
    return copies


def extract_single_feature(y, feat_kind, n_coeff, feat_format):
    """Extract a single feature representation from waveform y."""
    if feat_format == "baseline":
        return extract_baseline_mfcc(y, SR, n_coeff) if feat_kind == "mfcc" else extract_baseline_mfmc(y, SR, n_coeff)
    elif feat_format == "frames144":
        return extract_mfcc_frames144(y, SR, n_coeff) if feat_kind == "mfcc" else extract_mfmc_frames144(y, SR, n_coeff)
    elif feat_format == "stats144":
        return extract_mfcc_stats144(y, SR, n_coeff) if feat_kind == "mfcc" else extract_mfmc_stats144(y, SR, n_coeff)
    else:
        raise ValueError(f"Unknown format: {feat_format}")


def prepare_run_data(dataset, feat_kind, n_coeff, feat_format, setup, split_type, seed):
    """
    Prepare X_train, y_train, X_val, y_val, X_test, y_test with strict leak-free standardization.
    """
    meta = get_metadata(dataset)
    labels = np.load(FEAT_DIR / f"{dataset}_labels.npy")
    X_cached = np.load(FEAT_DIR / f"{dataset}_{feat_kind}_{n_coeff}_{feat_format}.npy")

    # Generate split
    if split_type == "stratified":
        df_split = make_stratified_split(meta, seed=seed)
    elif split_type == "speaker_independent":
        df_split = make_speaker_independent_split(meta, dataset, seed=seed)
    else:
        raise ValueError(f"Unknown split_type: {split_type}")

    tr_idx = df_split.index[df_split["split"] == "train"].values
    va_idx = df_split.index[df_split["split"] == "val"].values
    te_idx = df_split.index[df_split["split"] == "test"].values

    # Validation and test are ALWAYS unaugmented original clips
    X_val = X_cached[va_idx]
    y_val = labels[va_idx]
    X_test = X_cached[te_idx]
    y_test = labels[te_idx]

    if setup == "none":
        X_train = X_cached[tr_idx]
        y_train = labels[tr_idx]
    elif setup == "version_a":
        # Augment ONLY training clips (5x)
        rng = np.random.default_rng(seed)
        aug_feats = []
        aug_labels = []
        for idx in tr_idx:
            row = meta.iloc[idx]
            wav_path = ROOT_DIR / row["rel_path"]
            y_orig = load_audio_clip(wav_path, SR)
            aug_waves = augment_waveform(y_orig, rng)
            for y_aug in aug_waves:
                feat = extract_single_feature(y_aug, feat_kind, n_coeff, feat_format)
                aug_feats.append(feat)
                aug_labels.append(labels[idx])
        X_train = np.array(aug_feats, dtype=np.float32)
        y_train = np.array(aug_labels, dtype=np.int32)
    else:
        raise ValueError(f"Unknown setup: {setup}")

    # Standardization (strictly on train split only)
    scaler = StandardScaler()
    if feat_format in ["baseline", "stats144"]:
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        # Reshape to (N, D, 1) for Conv1D
        X_train_ready = np.expand_dims(X_train_scaled, axis=-1)
        X_val_ready = np.expand_dims(X_val_scaled, axis=-1)
        X_test_ready = np.expand_dims(X_test_scaled, axis=-1)
    elif feat_format == "frames144":
        # X shape: (N, 144, C)
        n_tr, t_steps, c_dim = X_train.shape
        n_va = X_val.shape[0]
        n_te = X_test.shape[0]

        scaler.fit(X_train.reshape(-1, c_dim))
        X_train_ready = scaler.transform(X_train.reshape(-1, c_dim)).reshape(n_tr, t_steps, c_dim)
        X_val_ready = scaler.transform(X_val.reshape(-1, c_dim)).reshape(n_va, t_steps, c_dim)
        X_test_ready = scaler.transform(X_test.reshape(-1, c_dim)).reshape(n_te, t_steps, c_dim)
    else:
        raise ValueError(f"Unknown feat_format: {feat_format}")

    return X_train_ready, y_train, X_val_ready, y_val, X_test_ready, y_test


def execute_run(
    dataset,
    feat_kind,
    n_coeff,
    feat_format,
    setup,
    split_type,
    seed,
    epochs=100,
    batch_size=32,
    lr=0.001,
    csv_out=RESULTS_DIR / "week4_results.csv",
):
    """Execute a single run, score once, and append to CSV."""
    set_seed(seed)
    num_classes_map = {"emodb": 7, "cafe": 7, "aesdd": 5, "mesd": 6}
    num_classes = num_classes_map[dataset]

    run_id = f"{dataset}_{feat_kind}_{n_coeff}_{feat_format}_{setup}_{split_type}_seed{seed}"
    model_ckpt_path = MODELS_DIR / f"ckpt_{run_id}.keras"

    print(f"\n[RUN] {run_id}")
    start_time = time.time()

    # 1. Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_run_data(
        dataset, feat_kind, n_coeff, feat_format, setup, split_type, seed
    )
    print(f"  Data shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # 2. Build model
    input_shape = X_train.shape[1:]
    model = build_1d_cnn(
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=lr,
        name=f"model_{run_id}",
    )

    # 3. Callbacks: Checkpoint best model by val_accuracy
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_ckpt_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=0,
        )
    ]

    # 4. Train
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    # 5. Load best validation checkpoint
    best_model = keras.models.load_model(str(model_ckpt_path))

    # 6. Evaluate
    val_loss, val_acc = best_model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
    train_loss, train_acc = best_model.evaluate(X_train, y_train, verbose=0)
    elapsed = time.time() - start_time

    print(f"  Finished in {elapsed:.1f}s | Val Acc: {val_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}% | Train Acc: {train_acc*100:.2f}%")

    # 7. Record row to CSV
    row = {
        "dataset": dataset,
        "feature": feat_kind,
        "coeffs": n_coeff,
        "format": feat_format,
        "setup": setup,
        "split_type": split_type,
        "seed": seed,
        "val_acc": round(float(val_acc), 4),
        "test_acc": round(float(test_acc), 4),
        "train_acc": round(float(train_acc), 4),
        "val_loss": round(float(val_loss), 4),
        "test_loss": round(float(test_loss), 4),
        "duration_s": round(elapsed, 1),
    }

    # Append safely
    csv_path = Path(csv_out)
    if csv_path.is_file():
        df_existing = pd.read_csv(csv_path)
        df_new = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)
    else:
        df_new = pd.DataFrame([row])
    df_new.to_csv(csv_path, index=False)

    return row


if __name__ == "__main__":
    # Test a single fast run on EmoDB
    res = execute_run(
        dataset="emodb",
        feat_kind="mfcc",
        n_coeff=12,
        feat_format="baseline",
        setup="none",
        split_type="stratified",
        seed=42,
        epochs=3,  # fast test
    )
    print("Test run successful:", res)
