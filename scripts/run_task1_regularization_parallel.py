"""
Task 1: Regularization Study on EmoDB (Parallel Runner).

Evaluates regularization techniques on EmoDB:
- reg_none (control)
- reg_l2 (L2 weight decay 1e-3 on dense layers)
- reg_spatial_dropout (SpatialDropout1D(0.2) in conv blocks)
- reg_early_stopping (EarlyStopping with patience=15)
- reg_cosine_lr (Cosine annealing learning rate schedule)
- reg_combined (L2 + SpatialDropout1D(0.15) + EarlyStopping)

Evaluated across:
- Splits: Stratified 60/20/20 and 5-fold Speaker-Disjoint
- Features: MFCC and MFMC
- Coefficients: 24 and 30
- Seeds: 42, 43, 44
- Setup: none (unaugmented baseline format)

Integrity rules:
- StandardScaler fitted strictly on training split only.
- Model selection strictly by best val_accuracy.
- Test scored once per run on frozen best weights.
- Appended to results/week5_results.csv.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import time
import gc
import random
import tempfile
from pathlib import Path
import concurrent.futures
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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


def worker_train_and_eval(task_dict):
    """
    Executed inside worker process.
    """
    import tensorflow as tf
    import keras
    from keras import layers

    # Limit threads per worker so 4 workers don't oversubscribe 12 cores
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)

    split_type = task_dict["split_type"]
    reg_config = task_dict["reg_config"]
    feat = task_dict["feat"]
    c = task_dict["c"]
    fold_idx = task_dict["fold"]
    seed = task_dict["seed"]
    X_feat = task_dict["X_feat"]
    y_all = task_dict["y_all"]
    meta = task_dict["meta"]
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.backend.clear_session()

    epochs = 100
    batch_size = 32
    num_classes = 7

    if split_type == "stratified":
        df_split = make_stratified_split(meta, seed=seed)
        tr_idx = df_split.index[df_split["split"] == "train"].values
        va_idx = df_split.index[df_split["split"] == "val"].values
        te_idx = df_split.index[df_split["split"] == "test"].values
        te_spk = np.nan
        va_spk = np.nan
    else:
        df_split = make_rotating_speaker_fold(meta, "emodb", fold_idx)
        tr_idx = df_split.index[df_split["split"] == "train"].values
        va_idx = df_split.index[df_split["split"] == "val"].values
        te_idx = df_split.index[df_split["split"] == "test"].values
        te_spk = ";".join(sorted(str(s) for s in set(df_split.loc[te_idx, "speaker_id"])))
        va_spk = ";".join(sorted(str(s) for s in set(df_split.loc[va_idx, "speaker_id"])))

    # Fit scaler strictly on training split
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_feat[tr_idx])[:, :, np.newaxis]
    X_va = scaler.transform(X_feat[va_idx])[:, :, np.newaxis]
    X_te = scaler.transform(X_feat[te_idx])[:, :, np.newaxis]
    y_tr = y_all[tr_idx]
    y_va = y_all[va_idx]
    y_te = y_all[te_idx]

    input_shape = (X_feat.shape[1], 1)
    use_l2 = reg_config in ["reg_l2", "reg_combined"]
    l2_reg = keras.regularizers.l2(1e-3) if use_l2 else None
    use_spatial_drop = reg_config in ["reg_spatial_dropout", "reg_combined"]
    spatial_rate = 0.15 if reg_config == "reg_combined" else 0.20

    inputs = layers.Input(shape=input_shape)
    
    # Block 1
    x = layers.Conv1D(8, 15, strides=1, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    if use_spatial_drop:
        x = layers.SpatialDropout1D(spatial_rate)(x)

    # Block 2
    x = layers.Conv1D(16, 15, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    if use_spatial_drop:
        x = layers.SpatialDropout1D(spatial_rate)(x)

    # Block 3
    x = layers.Conv1D(32, 15, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    if use_spatial_drop:
        x = layers.SpatialDropout1D(spatial_rate)(x)

    # Block 4
    x = layers.Conv1D(64, 15, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    if use_spatial_drop:
        x = layers.SpatialDropout1D(spatial_rate)(x)

    # Head
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(120, activation="relu", kernel_regularizer=l2_reg)(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", kernel_regularizer=l2_reg)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    steps_per_epoch = max(1, int(np.ceil(len(X_tr) / batch_size)))
    if reg_config in ["reg_cosine_lr", "reg_combined"]:
        total_steps = epochs * steps_per_epoch
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.001,
            decay_steps=total_steps,
            alpha=0.01
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    temp_id = f"t1_{split_type}_{reg_config}_{feat}_{c}_f{fold_idx}_s{seed}_{os.getpid()}"
    temp_ckpt = Path(tempfile.gettempdir()) / f"ckpt_{temp_id}.keras"

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
    model.fit(
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

    result_row = {
        "task": "task1_regularization",
        "dataset": "emodb",
        "feature": feat,
        "coeffs": c,
        "format": "baseline",
        "setup": "none",
        "reg_config": reg_config,
        "split_type": split_type,
        "fold": fold_idx if split_type == "speaker_independent" else np.nan,
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
    return result_row


def get_completed_keys():
    if not CSV_OUT.is_file():
        return set()
    try:
        df = pd.read_csv(CSV_OUT)
        t1 = df[df["task"] == "task1_regularization"]
        completed = set()
        for _, r in t1.iterrows():
            fold_val = int(r["fold"]) if pd.notna(r["fold"]) else -1
            key = (str(r["split_type"]), str(r["reg_config"]), str(r["feature"]), int(r["coeffs"]), fold_val, int(r["seed"]))
            completed.add(key)
        return completed
    except Exception as e:
        print("Warning reading CSV_OUT:", e)
        return set()


def run_task1_parallel(max_workers=4):
    print(f"=== Running Task 1: Regularization Grid (Workers: {max_workers}) ===")
    meta = get_metadata("emodb")
    cache_X = {}
    for feat in FEATURES:
        for c in COEFFS:
            cache_X[(feat, c)] = np.load(FEAT_DIR / f"emodb_{feat}_{c}_baseline.npy")
    y_all = np.load(FEAT_DIR / "emodb_labels.npy")

    completed = get_completed_keys()
    print(f"Completed Task 1 runs already recorded: {len(completed)}")

    tasks_to_run = []

    # 1. Stratified tasks
    for reg in CONFIGS:
        for feat in FEATURES:
            for c in COEFFS:
                for s in SEEDS:
                    key = ("stratified", reg, feat, c, -1, s)
                    if key not in completed:
                        tasks_to_run.append({
                            "split_type": "stratified",
                            "reg_config": reg,
                            "feat": feat,
                            "c": c,
                            "fold": -1,
                            "seed": s,
                            "X_feat": cache_X[(feat, c)],
                            "y_all": y_all,
                            "meta": meta
                        })

    # 2. Speaker-independent tasks
    for reg in CONFIGS:
        for feat in FEATURES:
            for c in COEFFS:
                for fold in range(5):
                    for s in SEEDS:
                        key = ("speaker_independent", reg, feat, c, fold, s)
                        if key not in completed:
                            tasks_to_run.append({
                                "split_type": "speaker_independent",
                                "reg_config": reg,
                                "feat": feat,
                                "c": c,
                                "fold": fold,
                                "seed": s,
                                "X_feat": cache_X[(feat, c)],
                                "y_all": y_all,
                                "meta": meta
                            })

    print(f"Total new runs to execute: {len(tasks_to_run)}")
    if len(tasks_to_run) == 0:
        print("All Task 1 runs are already complete!")
        return

    t_start = time.time()
    n_done = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(worker_train_and_eval, t): t for t in tasks_to_run}
        for future in concurrent.futures.as_completed(future_map):
            res = future.result()
            # Append cleanly to CSV in main process
            df_row = pd.DataFrame([res])
            if CSV_OUT.is_file():
                df_row.to_csv(CSV_OUT, mode="a", header=False, index=False)
            else:
                df_row.to_csv(CSV_OUT, index=False)
            
            n_done += 1
            if n_done % 15 == 0 or n_done == len(tasks_to_run):
                rate = (time.time() - t_start) / n_done
                remain = rate * (len(tasks_to_run) - n_done)
                print(f"[Task 1 Progress] {n_done}/{len(tasks_to_run)} done ({n_done/len(tasks_to_run)*100:.1f}%) | "
                      f"Last: {res['reg_config']} {res['feature']}-{res['coeffs']} {res['split_type']} | "
                      f"Val: {res['val_acc']*100:.1f}%, Test: {res['test_acc']*100:.1f}%, Gap: {res['train_val_gap']*100:.1f}% | "
                      f"Est remaining: {remain/60:.1f} min")

    total_time = time.time() - t_start
    print(f"=== Task 1 Finished in {total_time/60:.2f} minutes! ===")


if __name__ == "__main__":
    run_task1_parallel(max_workers=4)
