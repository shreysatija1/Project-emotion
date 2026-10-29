"""
Task 2: Acoustic Augmentations vs. Gaussian Noise (Parallel Runner).

Evaluates novel acoustic augmentations against clean Gaussian noise (Version A)
under rotating speaker-disjoint folds on EmoDB, AESDD, and CAFE:
- Augmentations:
  1. pitch_shift (±2 semitones: 0, -2, -1, +1, +2 semitones)
  2. time_stretch (0.9x to 1.1x: 1.0, 0.90, 0.95, 1.05, 1.10)
  3. spec_augment (time & frequency masking applied BEFORE mean+std pooling)
- Evaluated on:
  - EmoDB (5 folds, 7 classes)
  - AESDD (5 folds, 5 classes)
  - CAFE (6 folds, 7 classes)
- Features: MFCC and MFMC
- Coefficients: 24 and 30
- Seeds: 42, 43, 44
- Augmentation factor: 5x per training clip (identical to Version A)

Strict Integrity Guardrails:
- Augment training audio only, after the split. Val and test stay 100% clean.
- Zero speaker overlap: Train, Val, Test speakers are strictly disjoint.
- StandardScaler fitted strictly on training clips only.
- Model selected strictly by best validation accuracy on unseen validation speakers.
- Single-pass evaluation on unseen test speakers on frozen best checkpoint.
- Appended to results/week5_results.csv.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
import time
import gc
import random
import tempfile
import shutil
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

from data_loader import get_metadata, load_audio_clip, SR
from split_utils import make_rotating_speaker_fold
from extract_all_features import extract_baseline_mfcc, extract_baseline_mfmc
from acoustic_augment import augment_pitch_shift, augment_time_stretch, extract_spec_augment_features
from model import build_1d_cnn

DATASETS = ["emodb", "aesdd", "cafe"]
AUGS = ["pitch_shift", "time_stretch", "spec_augment"]
COEFFS = [24, 30]
FEATURES = ["mfcc", "mfmc"]
SEEDS = [42, 43, 44]

FOLDS_PER_DATASET = {
    "emodb": 5,
    "aesdd": 5,
    "cafe": 6
}
NUM_CLASSES_MAP = {
    "emodb": 7,
    "aesdd": 5,
    "cafe": 7
}


def build_and_cache_augmented_features(dataset, aug_type, feat_kind, n_coeff, fold_idx, seed):
    """
    Extract and cache 5x augmented features for training clips strictly.
    """
    cache_path = FEAT_DIR / f"cache_aug5_{dataset}_{aug_type}_{feat_kind}_{n_coeff}_fold{fold_idx}_seed{seed}.npz"
    if cache_path.is_file():
        loaded = np.load(cache_path, allow_pickle=True)
        return loaded["X"], loaded["y"]

    meta = get_metadata(dataset)
    df_split = make_rotating_speaker_fold(meta, dataset, fold_idx)
    tr_idx = df_split.index[df_split["split"] == "train"].values
    labels = np.load(FEAT_DIR / f"{dataset}_labels.npy")

    # Strict isolation assertions
    tr_spk = set(df_split.loc[tr_idx, "speaker_id"].unique())
    va_spk = set(df_split[df_split["split"] == "val"]["speaker_id"].unique())
    te_spk = set(df_split[df_split["split"] == "test"]["speaker_id"].unique())
    assert len(tr_spk & va_spk) == 0, f"Train-Val speaker overlap in {dataset} fold {fold_idx}"
    assert len(tr_spk & te_spk) == 0, f"Train-Test speaker overlap in {dataset} fold {fold_idx}"

    rng = np.random.default_rng(seed)
    aug_feats = []
    aug_labels = []

    for idx in tr_idx:
        row = meta.iloc[idx]
        wav_path = ROOT_DIR / row["rel_path"]
        y_orig = load_audio_clip(wav_path, SR)
        label = labels[idx]

        if aug_type == "pitch_shift":
            wav_copies = augment_pitch_shift(y_orig, SR, steps=(0, -2, -1, 1, 2))
            for y_c in wav_copies:
                if feat_kind == "mfcc":
                    feat = extract_baseline_mfcc(y_c, SR, n_coeff)
                else:
                    feat = extract_baseline_mfmc(y_c, SR, n_coeff)
                aug_feats.append(feat)
                aug_labels.append(label)

        elif aug_type == "time_stretch":
            wav_copies = augment_time_stretch(y_orig, rates=(1.0, 0.90, 0.95, 1.05, 1.10))
            for y_c in wav_copies:
                if feat_kind == "mfcc":
                    feat = extract_baseline_mfcc(y_c, SR, n_coeff)
                else:
                    feat = extract_baseline_mfmc(y_c, SR, n_coeff)
                aug_feats.append(feat)
                aug_labels.append(label)

        elif aug_type == "spec_augment":
            spec_copies = extract_spec_augment_features(y_orig, SR, feat_kind, n_coeff, rng, n_copies=5)
            for feat in spec_copies:
                aug_feats.append(feat)
                aug_labels.append(label)
        else:
            raise ValueError(f"Unknown aug_type: {aug_type}")

    X_train = np.array(aug_feats, dtype=np.float32)
    y_train = np.array(aug_labels, dtype=np.int32)
    np.savez_compressed(cache_path, X=X_train, y=y_train)
    return X_train, y_train


def worker_train_eval_aug(task_dict):
    """
    Executed inside worker process.
    """
    import tensorflow as tf
    import keras
    from keras import layers

    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)

    dataset = task_dict["dataset"]
    aug_type = task_dict["aug_type"]
    feat_kind = task_dict["feat_kind"]
    n_coeff = task_dict["n_coeff"]
    fold_idx = task_dict["fold"]
    seed = task_dict["seed"]

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.backend.clear_session()

    epochs = 100
    batch_size = 32
    num_classes = NUM_CLASSES_MAP[dataset]

    meta = get_metadata(dataset)
    df_split = make_rotating_speaker_fold(meta, dataset, fold_idx)
    va_idx = df_split.index[df_split["split"] == "val"].values
    te_idx = df_split.index[df_split["split"] == "test"].values

    te_spk = ";".join(sorted(str(s) for s in set(df_split.loc[te_idx, "speaker_id"])))
    va_spk = ";".join(sorted(str(s) for s in set(df_split.loc[va_idx, "speaker_id"])))

    # Clean unaugmented validation and test data
    X_all = np.load(FEAT_DIR / f"{dataset}_{feat_kind}_{n_coeff}_baseline.npy")
    labels = np.load(FEAT_DIR / f"{dataset}_labels.npy")

    X_val = X_all[va_idx]
    y_val = labels[va_idx]
    X_test = X_all[te_idx]
    y_test = labels[te_idx]

    # Load / extract 5x augmented training data
    X_train, y_train = build_and_cache_augmented_features(dataset, aug_type, feat_kind, n_coeff, fold_idx, seed)

    # Standardize strictly on training stats
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)[:, :, np.newaxis]
    X_val_scaled = scaler.transform(X_val)[:, :, np.newaxis]
    X_test_scaled = scaler.transform(X_test)[:, :, np.newaxis]

    model = build_1d_cnn(
        input_shape=X_train_scaled.shape[1:],
        num_classes=num_classes,
        learning_rate=0.001,
        name=f"model_{dataset}_{aug_type}_{feat_kind}_{n_coeff}_f{fold_idx}_s{seed}"
    )

    temp_id = f"t2_{dataset}_{aug_type}_{feat_kind}_{n_coeff}_f{fold_idx}_s{seed}_{os.getpid()}"
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

    t0 = time.time()
    model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )

    best_model = keras.models.load_model(str(temp_ckpt))
    val_loss, val_acc = best_model.evaluate(X_val_scaled, y_val, verbose=0)
    test_loss, test_acc = best_model.evaluate(X_test_scaled, y_test, verbose=0)
    train_loss, train_acc = best_model.evaluate(X_train_scaled, y_train, verbose=0)
    elapsed = round(time.time() - t0, 2)
    temp_ckpt.unlink(missing_ok=True)

    result_row = {
        "task": "task2_augmentation",
        "dataset": dataset,
        "feature": feat_kind,
        "coeffs": n_coeff,
        "format": "baseline",
        "setup": aug_type,
        "reg_config": "standard",
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
    return result_row


def get_completed_keys_task2():
    if not CSV_OUT.is_file():
        return set()
    try:
        df = pd.read_csv(CSV_OUT)
        t2 = df[df["task"] == "task2_augmentation"]
        completed = set()
        for _, r in t2.iterrows():
            key = (str(r["dataset"]), str(r["setup"]), str(r["feature"]), int(r["coeffs"]), int(r["fold"]), int(r["seed"]))
            completed.add(key)
        return completed
    except Exception as e:
        print("Warning reading completed task2 keys:", e)
        return set()


def run_task2_parallel(max_workers=4):
    print(f"=== Running Task 2: Acoustic Augmentations Grid (Workers: {max_workers}) ===")
    completed = get_completed_keys_task2()
    print(f"Completed Task 2 runs already recorded: {len(completed)}")

    tasks_to_run = []
    for d in DATASETS:
        n_folds = FOLDS_PER_DATASET[d]
        for aug in AUGS:
            for feat in FEATURES:
                for c in COEFFS:
                    for fold in range(n_folds):
                        for s in SEEDS:
                            key = (d, aug, feat, c, fold, s)
                            if key not in completed:
                                tasks_to_run.append({
                                    "dataset": d,
                                    "aug_type": aug,
                                    "feat_kind": feat,
                                    "n_coeff": c,
                                    "fold": fold,
                                    "seed": s
                                })

    print(f"Total new Task 2 runs to execute: {len(tasks_to_run)}")
    if len(tasks_to_run) == 0:
        print("All Task 2 runs are already complete!")
        return

    t_start = time.time()
    n_done = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(worker_train_eval_aug, t): t for t in tasks_to_run}
        for future in concurrent.futures.as_completed(future_map):
            res = future.result()
            df_row = pd.DataFrame([res])
            if CSV_OUT.is_file():
                df_row.to_csv(CSV_OUT, mode="a", header=False, index=False)
            else:
                df_row.to_csv(CSV_OUT, index=False)

            n_done += 1
            if n_done % 15 == 0 or n_done == len(tasks_to_run):
                rate = (time.time() - t_start) / n_done
                remain = rate * (len(tasks_to_run) - n_done)
                print(f"[Task 2 Progress] {n_done}/{len(tasks_to_run)} done ({n_done/len(tasks_to_run)*100:.1f}%) | "
                      f"Last: {res['dataset'].upper()} {res['setup']} {res['feature']}-{res['coeffs']} fold{res['fold']} | "
                      f"Val: {res['val_acc']*100:.1f}%, Test: {res['test_acc']*100:.1f}%, Gap: {res['train_val_gap']*100:.1f}% | "
                      f"Est remaining: {remain/60:.1f} min")

    total_time = time.time() - t_start
    print(f"=== Task 2 Finished in {total_time/60:.2f} minutes! ===")


if __name__ == "__main__":
    run_task2_parallel(max_workers=4)
