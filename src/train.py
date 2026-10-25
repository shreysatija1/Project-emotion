"""
Training script for Speech Emotion Recognition 1D-CNN.

Reproducing:
"Data augmentation using a 1D-CNN model with MFCC/MFMC features for speech emotion recognition"
Automatika (2024).

Modes:
  - none: Original train clips only (321 clips, Week 2 split). Val/test are original 107 clips each.
  - after_split: Version A (train augmented 5x to 1605 clips; val/test are original 107 clips each).
  - before_split: Version B (all 535 clips augmented 5x to 2675 clips, then split 60/20/20).

Training Rules:
  - Standardize features using train statistics ONLY (StandardScaler fitted strictly on X_train).
  - Training strictly evaluates on train and validation sets. NO TEST EVALUATION.
  - Checkpoints best model by validation accuracy (val_accuracy).
  - Saves per-epoch metrics to CSV.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras

from config import ROOT, RAW_DIR, PROC_DIR, FEAT_DIR, RESULTS_DIR
from cache import load_features, load_split
from model import build_1d_cnn, prepare_input


def set_seed(seed):
    """Set all random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data_by_mode(feature_type, n_coeff, aug_mode):
    """
    Load data corresponding to the requested augmentation mode.
    Uses canonical naming:
      - none: features/emodb_{kind}_{n}.npy and data/processed/splits.csv
      - after_split: features/version_a_{kind}_{n}.npy and data/processed/version_a_metadata.csv
      - before_split: features/version_b_{kind}_{n}.npy and data/processed/version_b_metadata.csv

    Returns
    -------
    X_train, y_train, X_val, y_val, (meta_train, meta_val)
    """
    kind = feature_type.lower().strip()
    mode = aug_mode.lower().strip()

    if mode == "none":
        X = load_features(kind, n_coeff)
        tr_idx, va_idx, te_idx, y = load_split()
        meta = pd.read_csv(PROC_DIR / "splits.csv")
        X_train, y_train = X[tr_idx], y[tr_idx]
        X_val, y_val = X[va_idx], y[va_idx]
        meta_train = meta.iloc[tr_idx].copy()
        meta_val = meta.iloc[va_idx].copy()

    elif mode == "after_split":
        feat_path = FEAT_DIR / f"version_a_{kind}_{n_coeff}.npy"
        label_path = FEAT_DIR / "version_a_labels.npy"
        meta_path = PROC_DIR / "version_a_metadata.csv"
        if not meta_path.is_file():
            meta_path = FEAT_DIR / "version_a_metadata.csv"

        X = np.load(feat_path)
        y = np.load(label_path)
        meta = pd.read_csv(meta_path)

        tr_idx = meta.index[meta["split"] == "train"].values
        va_idx = meta.index[meta["split"] == "val"].values

        X_train, y_train = X[tr_idx], y[tr_idx]
        X_val, y_val = X[va_idx], y[va_idx]
        meta_train = meta.iloc[tr_idx].copy()
        meta_val = meta.iloc[va_idx].copy()

    elif mode == "before_split":
        feat_path = FEAT_DIR / f"version_b_{kind}_{n_coeff}.npy"
        label_path = FEAT_DIR / "version_b_labels.npy"
        meta_path = PROC_DIR / "version_b_metadata.csv"
        if not meta_path.is_file():
            meta_path = FEAT_DIR / "version_b_metadata.csv"

        X = np.load(feat_path)
        y = np.load(label_path)
        meta = pd.read_csv(meta_path)

        tr_idx = meta.index[meta["split"] == "train"].values
        va_idx = meta.index[meta["split"] == "val"].values

        X_train, y_train = X[tr_idx], y[tr_idx]
        X_val, y_val = X[va_idx], y[va_idx]
        meta_train = meta.iloc[tr_idx].copy()
        meta_val = meta.iloc[va_idx].copy()

    else:
        raise ValueError(f"Unknown aug_mode: '{aug_mode}'. Must be 'none', 'after_split', or 'before_split'.")

    return X_train, y_train, X_val, y_val, (meta_train, meta_val)


def train_run(
    feature_type,
    n_coeff,
    aug_mode,
    seed=42,
    epochs=100,
    batch_size=32,
    lr=0.001,
    early_stopping=False,
    patience=20,
    results_dir=RESULTS_DIR,
    models_dir=ROOT / "models",
):
    """
    Execute a single training run without test-set evaluation.
    """
    set_seed(seed)
    results_dir = Path(results_dir)
    models_dir = Path(models_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"{feature_type}_{n_coeff}_{aug_mode}_seed{seed}"
    best_model_path = models_dir / f"best_model_{run_name}.keras"
    csv_log_path = results_dir / f"history_{run_name}.csv"
    scaler_path = results_dir / f"scaler_{run_name}.joblib"
    summary_path = results_dir / f"summary_{run_name}.json"

    print(f"\n[START RUN] {run_name}")
    print(f"  Configuration: feature={feature_type}, n={n_coeff}, mode={aug_mode}, seed={seed}")
    print(f"  Training: epochs={epochs}, batch_size={batch_size}, lr={lr}, early_stop={early_stopping}")

    start_time = time.time()

    # 1. Load Data
    X_train, y_train, X_val, y_val, (meta_train, meta_val) = load_data_by_mode(
        feature_type, n_coeff, aug_mode
    )
    print(f"  Data loaded: Train shape={X_train.shape}, Val shape={X_val.shape}")

    # 2. Standardize with Train Statistics ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    joblib.dump(scaler, scaler_path)

    # 3. Reshape for Conv1D: (N, D, 1)
    X_train_3d = prepare_input(X_train_scaled)
    X_val_3d = prepare_input(X_val_scaled)

    # 4. Build Model
    input_dim = 2 * n_coeff
    model = build_1d_cnn(
        input_dim=input_dim,
        num_classes=7,
        filters=(8, 16, 32, 64),
        kernel_size=15,
        dense_units=120,
        dropout_conv=0.3,
        dropout_dense=0.2,
        learning_rate=lr,
        name=f"ser_{run_name}",
    )

    # 5. Callbacks
    callbacks = [
        keras.callbacks.CSVLogger(filename=str(csv_log_path)),
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=0,
        ),
    ]
    if early_stopping:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=0,
            )
        )

    # 6. Fit Model (NO TEST EVALUATION)
    history = model.fit(
        X_train_3d,
        y_train,
        validation_data=(X_val_3d, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    elapsed_s = round(time.time() - start_time, 2)

    # 7. Load Best Model and evaluate validation performance strictly
    if best_model_path.exists():
        best_model = keras.models.load_model(str(best_model_path))
    else:
        best_model = model

    val_loss, val_acc = best_model.evaluate(X_val_3d, y_val, verbose=0)
    train_loss, train_acc = best_model.evaluate(X_train_3d, y_train, verbose=0)

    summary = {
        "run_name": run_name,
        "feature_type": feature_type,
        "n_coeff": n_coeff,
        "aug_mode": aug_mode,
        "seed": seed,
        "epochs_trained": len(history.history["loss"]),
        "train_loss": round(float(train_loss), 4),
        "train_accuracy": round(float(train_acc), 4),
        "val_loss": round(float(val_loss), 4),
        "val_accuracy": round(float(val_acc), 4),
        "elapsed_seconds": elapsed_s,
        "best_model_path": str(best_model_path),
        "scaler_path": str(scaler_path),
        "history_path": str(csv_log_path),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[FINISHED RUN] {run_name} in {elapsed_s}s | Best Val Acc: {val_acc:.4f} | Train Acc: {train_acc:.4f}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train 1D-CNN SER Model")
    parser.add_argument("--feature_type", choices=["mfcc", "mfmc"], required=True)
    parser.add_argument("--coefficients", "-n", type=int, choices=[12, 24, 30], required=True)
    parser.add_argument("--aug_mode", choices=["none", "after_split", "before_split"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--results_dir", default=str(RESULTS_DIR))
    parser.add_argument("--models_dir", default=str(ROOT / "models"))

    args = parser.parse_args()

    train_run(
        feature_type=args.feature_type,
        n_coeff=args.coefficients,
        aug_mode=args.aug_mode,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        early_stopping=args.early_stopping,
        patience=args.patience,
        results_dir=args.results_dir,
        models_dir=args.models_dir,
    )


if __name__ == "__main__":
    main()
