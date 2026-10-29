"""
Task 1: Regularization Study on EmoDB.

Evaluates whether regularization techniques narrow the generalization gap on EmoDB:
- Options:
  1. reg_none (baseline control)
  2. reg_l2 (L2 weight decay 1e-3 on dense layers)
  3. reg_spatial_dropout (SpatialDropout1D(0.2) in conv blocks)
  4. reg_early_stopping (EarlyStopping with patience=15)
  5. reg_cosine_lr (Cosine-annealed learning rate)
  6. reg_combined (L2 + SpatialDropout1D + EarlyStopping)

Evaluated across:
- Features: MFCC, MFMC
- Coefficients: 24, 30
- Splits: Stratified 60/20/20 and 5-fold Speaker-Disjoint
- Seeds: 42, 43, 44
- Setup: none (unaugmented baseline format)

Strict Protocol:
- StandardScaler fitted strictly on train split only.
- Model checkpoint / selection strictly by best val_accuracy.
- Test scored once per run on frozen best weights.
- Output appended to results/week5_results.csv.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import time
import gc
import random
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras
from keras import layers

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
FEAT_DIR = ROOT_DIR / "features"
RESULTS_DIR = ROOT_DIR / "results"
CSV_OUT = RESULTS_DIR / "week5_results.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import get_metadata
from split_utils import make_stratified_split, make_rotating_speaker_fold

CONFIGS = [
    "reg_none",
    "reg_l2",
    "reg_spatial_dropout",
    "reg_early_stopping",
    "reg_cosine_lr",
    "reg_combined"
]
COEFFS = [24, 30]
FEATURES = ["mfcc", "mfmc"]
SEEDS = [42, 43, 44]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_regularized_model(input_dim, num_classes, reg_config, total_epochs=100, steps_per_epoch=10):
    input_shape = (input_dim, 1)
    
    use_l2 = reg_config in ["reg_l2", "reg_combined"]
    l2_reg = keras.regularizers.l2(1e-3) if use_l2 else None
    
    use_spatial_drop = reg_config in ["reg_spatial_dropout", "reg_combined"]
    spatial_rate = 0.15 if reg_config == "reg_combined" else 0.20

    inputs = layers.Input(shape=input_shape, name="input_features")
    
    # Block 1
    x = layers.Conv1D(8, 15, strides=1, padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)
    if use_spatial_drop:
        x = layers.SpatialDropout1D(spatial_rate, name="spdrop1")(x)
        
    # Block 2
    x = layers.Conv1D(16, 15, strides=1, padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)
    if use_spatial_drop:
        x = layers.SpatialDropout1D(spatial_rate, name="spdrop2")(x)
        
    # Block 3
    x = layers.Conv1D(32, 15, strides=1, padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool3")(x)
    if use_spatial_drop:
        x = layers.SpatialDropout1D(spatial_rate, name="spdrop3")(x)
        
    # Block 4
    x = layers.Conv1D(64, 15, strides=1, padding="same", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool4")(x)
    if use_spatial_drop:
        x = layers.SpatialDropout1D(spatial_rate, name="spdrop4")(x)
        
    # Dense head
    x = layers.Dropout(0.3, name="dropout_conv")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(120, activation="relu", kernel_regularizer=l2_reg, name="dense120")(x)
    x = layers.Dropout(0.2, name="dropout_dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", kernel_regularizer=l2_reg, name="emotion_output")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name=f"model_{reg_config}")
    
    # Optimizer / Learning rate
    if reg_config in ["reg_cosine_lr", "reg_combined"]:
        total_steps = total_epochs * steps_per_epoch
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.001,
            decay_steps=total_steps,
            alpha=0.01
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def get_completed_runs():
    if not CSV_OUT.is_file():
        return set()
    try:
        df = pd.read_csv(CSV_OUT)
        t1 = df[df["task"] == "task1_regularization"]
        completed = set()
        for _, r in t1.iterrows():
            key = (
                str(r["split_type"]),
                str(r["reg_config"]),
                str(r["feature"]),
                int(r["coeffs"]),
                int(r["fold"]) if pd.notna(r["fold"]) else -1,
                int(r["seed"])
            )
            completed.add(key)
        return completed
    except Exception as e:
        print("Error reading completed runs:", e)
        return set()


def run_task1():
    print("=== Starting Task 1: Regularization Study on EmoDB ===")
    meta = get_metadata("emodb")
    num_classes = 7
    epochs = 100
    batch_size = 32
    
    # Pre-load raw baseline features for EmoDB
    cache_X = {}
    for feat in FEATURES:
        for c in COEFFS:
            cache_X[(feat, c)] = np.load(FEAT_DIR / f"emodb_{feat}_{c}_baseline.npy")
    y_all = np.load(FEAT_DIR / "emodb_labels.npy")
    
    completed = get_completed_runs()
    print(f"Already completed runs in {CSV_OUT}: {len(completed)}")
    
    total_runs = (
        len(CONFIGS) * len(FEATURES) * len(COEFFS) * len(SEEDS) + # Stratified: 6 * 2 * 2 * 3 = 72
        len(CONFIGS) * len(FEATURES) * len(COEFFS) * 5 * len(SEEDS) # Speaker-disjoint: 6 * 2 * 2 * 5 * 3 = 360
    )
    print(f"Total Task 1 runs to execute: {total_runs}")
    
    run_count = 0
    start_total = time.time()
    
    # 1. Stratified Runs
    for reg_config in CONFIGS:
        for feat in FEATURES:
            for c in COEFFS:
                X_feat = cache_X[(feat, c)]
                for seed in SEEDS:
                    key = ("stratified", reg_config, feat, c, -1, seed)
                    if key in completed:
                        continue
                        
                    set_seed(seed)
                    keras.backend.clear_session()
                    
                    # Split
                    df_split = make_stratified_split(meta, seed=seed)
                    tr_idx = df_split.index[df_split["split"] == "train"].values
                    va_idx = df_split.index[df_split["split"] == "val"].values
                    te_idx = df_split.index[df_split["split"] == "test"].values
                    
                    # Standardize strictly on train
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_feat[tr_idx])[:, :, np.newaxis]
                    X_va = scaler.transform(X_feat[va_idx])[:, :, np.newaxis]
                    X_te = scaler.transform(X_feat[te_idx])[:, :, np.newaxis]
                    y_tr = y_all[tr_idx]
                    y_va = y_all[va_idx]
                    y_te = y_all[te_idx]
                    
                    steps_per_epoch = max(1, int(np.ceil(len(X_tr) / batch_size)))
                    model = build_regularized_model(
                        input_dim=X_feat.shape[1],
                        num_classes=num_classes,
                        reg_config=reg_config,
                        total_epochs=epochs,
                        steps_per_epoch=steps_per_epoch
                    )
                    
                    temp_ckpt = Path(tempfile.gettempdir()) / f"ckpt_t1_strat_{reg_config}_{feat}_{c}_s{seed}.keras"
                    callbacks = [
                        keras.callbacks.ModelCheckpoint(
                            filepath=str(temp_ckpt),
                            monitor="val_accuracy",
                            save_best_only=True,
                            mode="max",
                            verbose=0
                        )
                    ]
                    if reg_config in ["reg_early_stopping", "reg_combined"]:
                        callbacks.append(
                            keras.callbacks.EarlyStopping(
                                monitor="val_accuracy",
                                patience=15,
                                mode="max",
                                restore_best_weights=True,
                                verbose=0
                            )
                        )
                        
                    t0 = time.time()
                    history = model.fit(
                        X_tr, y_tr,
                        validation_data=(X_va, y_va),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=0
                    )
                    
                    best_model = keras.models.load_model(str(temp_ckpt))
                    val_loss, val_acc = best_model.evaluate(X_va, y_va, verbose=0)
                    test_loss, test_acc = best_model.evaluate(X_te, y_te, verbose=0)
                    train_loss, train_acc = best_model.evaluate(X_tr, y_tr, verbose=0)
                    elapsed = round(time.time() - t0, 2)
                    temp_ckpt.unlink(missing_ok=True)
                    
                    row = {
                        "task": "task1_regularization",
                        "dataset": "emodb",
                        "feature": feat,
                        "coeffs": c,
                        "format": "baseline",
                        "setup": "none",
                        "reg_config": reg_config,
                        "split_type": "stratified",
                        "fold": np.nan,
                        "seed": seed,
                        "test_spk": np.nan,
                        "val_spk": np.nan,
                        "val_acc": round(float(val_acc), 4),
                        "test_acc": round(float(test_acc), 4),
                        "train_acc": round(float(train_acc), 4),
                        "val_loss": round(float(val_loss), 4),
                        "test_loss": round(float(test_loss), 4),
                        "train_val_gap": round(float(train_acc - val_acc), 4),
                        "duration_s": elapsed,
                    }
                    
                    df_row = pd.DataFrame([row])
                    if CSV_OUT.is_file():
                        df_row.to_csv(CSV_OUT, mode="a", header=False, index=False)
                    else:
                        df_row.to_csv(CSV_OUT, index=False)
                        
                    run_count += 1
                    completed.add(key)
                    if run_count % 12 == 0 or run_count == 1:
                        print(f"[Stratified] Run {run_count} | {reg_config} {feat}-{c} s{seed} | Val: {val_acc*100:.1f}% | Test: {test_acc*100:.1f}% | Train-Val Gap: {(train_acc-val_acc)*100:.1f}% | {elapsed}s")

    # 2. Speaker-Disjoint Runs
    for reg_config in CONFIGS:
        for feat in FEATURES:
            for c in COEFFS:
                X_feat = cache_X[(feat, c)]
                for fold_idx in range(5):
                    df_split = make_rotating_speaker_fold(meta, "emodb", fold_idx)
                    tr_idx = df_split.index[df_split["split"] == "train"].values
                    va_idx = df_split.index[df_split["split"] == "val"].values
                    te_idx = df_split.index[df_split["split"] == "test"].values
                    
                    te_spk = ";".join(sorted(str(s) for s in set(df_split.loc[te_idx, "speaker_id"])))
                    va_spk = ";".join(sorted(str(s) for s in set(df_split.loc[va_idx, "speaker_id"])))
                    
                    # Standardize strictly on train
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_feat[tr_idx])[:, :, np.newaxis]
                    X_va = scaler.transform(X_feat[va_idx])[:, :, np.newaxis]
                    X_te = scaler.transform(X_feat[te_idx])[:, :, np.newaxis]
                    y_tr = y_all[tr_idx]
                    y_va = y_all[va_idx]
                    y_te = y_all[te_idx]
                    
                    for seed in SEEDS:
                        key = ("speaker_independent", reg_config, feat, c, fold_idx, seed)
                        if key in completed:
                            continue
                            
                        set_seed(seed)
                        keras.backend.clear_session()
                        
                        steps_per_epoch = max(1, int(np.ceil(len(X_tr) / batch_size)))
                        model = build_regularized_model(
                            input_dim=X_feat.shape[1],
                            num_classes=num_classes,
                            reg_config=reg_config,
                            total_epochs=epochs,
                            steps_per_epoch=steps_per_epoch
                        )
                        
                        temp_ckpt = Path(tempfile.gettempdir()) / f"ckpt_t1_spk_{reg_config}_{feat}_{c}_f{fold_idx}_s{seed}.keras"
                        callbacks = [
                            keras.callbacks.ModelCheckpoint(
                                filepath=str(temp_ckpt),
                                monitor="val_accuracy",
                                save_best_only=True,
                                mode="max",
                                verbose=0
                            )
                        ]
                        if reg_config in ["reg_early_stopping", "reg_combined"]:
                            callbacks.append(
                                keras.callbacks.EarlyStopping(
                                    monitor="val_accuracy",
                                    patience=15,
                                    mode="max",
                                    restore_best_weights=True,
                                    verbose=0
                                )
                            )
                            
                        t0 = time.time()
                        history = model.fit(
                            X_tr, y_tr,
                            validation_data=(X_va, y_va),
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=callbacks,
                            verbose=0
                        )
                        
                        best_model = keras.models.load_model(str(temp_ckpt))
                        val_loss, val_acc = best_model.evaluate(X_va, y_va, verbose=0)
                        test_loss, test_acc = best_model.evaluate(X_te, y_te, verbose=0)
                        train_loss, train_acc = best_model.evaluate(X_tr, y_tr, verbose=0)
                        elapsed = round(time.time() - t0, 2)
                        temp_ckpt.unlink(missing_ok=True)
                        
                        row = {
                            "task": "task1_regularization",
                            "dataset": "emodb",
                            "feature": feat,
                            "coeffs": c,
                            "format": "baseline",
                            "setup": "none",
                            "reg_config": reg_config,
                            "split_type": "speaker_independent",
                            "fold": fold_idx,
                            "seed": seed,
                            "test_spk": te_spk,
                            "val_spk": va_spk,
                            "val_acc": round(float(val_acc), 4),
                            "test_acc": round(float(test_acc), 4),
                            "train_acc": round(float(train_acc), 4),
                            "val_loss": round(float(val_loss), 4),
                            "test_loss": round(float(test_loss), 4),
                            "train_val_gap": round(float(train_acc - val_acc), 4),
                            "duration_s": elapsed,
                        }
                        
                        df_row = pd.DataFrame([row])
                        if CSV_OUT.is_file():
                            df_row.to_csv(CSV_OUT, mode="a", header=False, index=False)
                        else:
                            df_row.to_csv(CSV_OUT, index=False)
                            
                        run_count += 1
                        completed.add(key)
                        if run_count % 20 == 0:
                            print(f"[Spk-Indep] Run {run_count} | {reg_config} {feat}-{c} fold{fold_idx} s{seed} | Val: {val_acc*100:.1f}% | Test: {test_acc*100:.1f}% | Gap: {(train_acc-val_acc)*100:.1f}% | {elapsed}s")

    total_time = time.time() - start_total
    print(f"=== Task 1 Complete! Executed {run_count} runs in {total_time/60:.2f} minutes ===")

if __name__ == "__main__":
    run_task1()
