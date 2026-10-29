"""
Task 3 Execution Engine: Rotating Speaker-Disjoint Folds.

Reproducing Thomas Mary Little Flower et al. (Automatika, 2024).
Evaluation Protocol:
- Datasets: EmoDB (5 folds), AESDD (5 folds), CAFE (6 folds)
- Coefficients: 24 and 30 only
- Features: MFCC and MFMC
- Setups: none (clean train only) and version_a (clean 5x augmentation on train only)
- Seeds: 42, 43, 44
- Total runs: (5 + 5 + 6) * 2 * 2 * 2 * 3 = 384 runs.

Strict Integrity Rules:
1. Zero speaker overlap: Train, Val, and Test speakers are 100% disjoint.
2. Augmentation and feature caches are keyed by dataset, fold, and seed, and built from that fold's train clips only.
3. StandardScaler is fitted strictly on training clips only.
4. Model selected strictly by best validation accuracy on unseen validation speakers.
5. Single-pass evaluation on unseen test speakers on frozen best checkpoint.
6. Execution order: EmoDB -> AESDD -> CAFE without pausing.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
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

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
FEAT_DIR = ROOT_DIR / "features"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models" / "week4_task3"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import get_metadata, load_audio_clip, SR
from split_utils import make_rotating_speaker_fold
from features_144 import extract_raw_mfcc, extract_raw_mfmc
from extract_all_features import extract_baseline_mfcc, extract_baseline_mfmc
from model import build_1d_cnn

CSV_OUT = RESULTS_DIR / "week4_results.csv"

COEFFS = [24, 30]
FEATURES = ["mfcc", "mfmc"]
SETUPS = ["none", "version_a"]
SEEDS = [42, 43, 44]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def augment_waveform(y, rng):
    noise_std = [0.0, 0.0, 0.01, 0.025, 0.05]
    copies = []
    for sigma in noise_std:
        if sigma == 0.0:
            copies.append(y.copy())
        else:
            noise = rng.normal(0.0, sigma, size=y.shape).astype(np.float32)
            copies.append(y + noise)
    return copies


def prepare_task3_fold_data(dataset, feat_kind, n_coeff, setup, fold_idx, seed):
    """
    Prepare data for a specific rotating speaker fold with complete isolation.
    """
    meta = get_metadata(dataset)
    df_split = make_rotating_speaker_fold(meta, dataset, fold_idx)

    # 1. Strict assertions on speaker disjointness
    tr_spk = set(df_split[df_split["split"] == "train"]["speaker_id"].unique())
    va_spk = set(df_split[df_split["split"] == "val"]["speaker_id"].unique())
    te_spk = set(df_split[df_split["split"] == "test"]["speaker_id"].unique())

    assert len(tr_spk & va_spk) == 0, f"FATAL: Train & Val overlap: {tr_spk & va_spk}"
    assert len(tr_spk & te_spk) == 0, f"FATAL: Train & Test overlap: {tr_spk & te_spk}"
    assert len(va_spk & te_spk) == 0, f"FATAL: Val & Test overlap: {va_spk & te_spk}"

    tr_idx = df_split.index[df_split["split"] == "train"].values
    va_idx = df_split.index[df_split["split"] == "val"].values
    te_idx = df_split.index[df_split["split"] == "test"].values

    # Check that each sample in train index belongs strictly to tr_spk
    for idx in tr_idx:
        spk = df_split.iloc[idx]["speaker_id"]
        assert spk in tr_spk, f"FATAL: tr_idx {idx} speaker {spk} not in tr_spk!"
        assert spk not in va_spk, f"FATAL: tr_idx {idx} speaker {spk} in va_spk!"
        assert spk not in te_spk, f"FATAL: tr_idx {idx} speaker {spk} in te_spk!"

    # 2. Load unaugmented feature vectors
    X_all = np.load(FEAT_DIR / f"{dataset}_{feat_kind}_{n_coeff}_baseline.npy")
    labels = np.load(FEAT_DIR / f"{dataset}_labels.npy")

    X_val = X_all[va_idx]
    y_val = labels[va_idx]
    X_test = X_all[te_idx]
    y_test = labels[te_idx]

    # 3. Train data preparation
    if setup == "none":
        X_train = X_all[tr_idx]
        y_train = labels[tr_idx]
    elif setup == "version_a":
        # Augmentation cache keyed by dataset, fold, and seed
        aug_cache = FEAT_DIR / f"cache_aug_{dataset}_{feat_kind}_{n_coeff}_fold{fold_idx}_seed{seed}.npz"
        if aug_cache.is_file():
            loaded = np.load(aug_cache, allow_pickle=True)
            X_train = loaded["X"]
            y_train = loaded["y"]
            aug_spks = loaded["spks"]
            # Strict verification of cached samples
            assert len(set(aug_spks) & va_spk) == 0, f"Cached aug has val speakers: {set(aug_spks) & va_spk}"
            assert len(set(aug_spks) & te_spk) == 0, f"Cached aug has test speakers: {set(aug_spks) & te_spk}"
        else:
            rng = np.random.default_rng(seed)
            aug_feats = []
            aug_labels = []
            aug_spks = []
            for idx in tr_idx:
                row = meta.iloc[idx]
                spk = row["speaker_id"]
                assert spk in tr_spk and spk not in va_spk and spk not in te_spk
                wav_path = ROOT_DIR / row["rel_path"]
                y_orig = load_audio_clip(wav_path, SR)
                aug_waves = augment_waveform(y_orig, rng)
                for y_aug in aug_waves:
                    if feat_kind == "mfcc":
                        feat = extract_baseline_mfcc(y_aug, SR, n_coeff)
                    else:
                        feat = extract_baseline_mfmc(y_aug, SR, n_coeff)
                    aug_feats.append(feat)
                    aug_labels.append(labels[idx])
                    aug_spks.append(spk)

            X_train = np.array(aug_feats, dtype=np.float32)
            y_train = np.array(aug_labels, dtype=np.int32)
            np.savez_compressed(aug_cache, X=X_train, y=y_train, spks=np.array(aug_spks))

            # Verify no training sample (original or augmented) comes from val or test
            assert len(set(aug_spks) & va_spk) == 0, "FATAL: Augmented sample contains validation speaker!"
            assert len(set(aug_spks) & te_spk) == 0, "FATAL: Augmented sample contains test speaker!"

    # 4. Standardize using training statistics ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 5. Reshape for 1D-CNN: (N, D, 1)
    X_train_3d = X_train_scaled[:, :, np.newaxis]
    X_val_3d = X_val_scaled[:, :, np.newaxis]
    X_test_3d = X_test_scaled[:, :, np.newaxis]

    test_spk_str = ";".join(sorted(str(s) for s in te_spk))
    val_spk_str = ";".join(sorted(str(s) for s in va_spk))

    return X_train_3d, y_train, X_val_3d, y_val, X_test_3d, y_test, test_spk_str, val_spk_str


COMPLETED_SET = set()


def refresh_completed_set():
    global COMPLETED_SET
    if CSV_OUT.is_file():
        try:
            df = pd.read_csv(CSV_OUT)
            t3 = df[df["split_type"] == "speaker_independent"]
            COMPLETED_SET = set(
                (str(r["dataset"]).lower(), str(r["feature"]).lower(), int(r["coeffs"]), str(r["setup"]).lower(), int(r["fold"]), int(r["seed"]))
                for _, r in t3.iterrows()
                if pd.notna(r["fold"]) and pd.notna(r["seed"])
            )
        except Exception as e:
            print("Warning reading CSV for completed set:", e)


def is_already_run(dataset, feat, coeffs, setup, fold, seed):
    key = (str(dataset).lower(), str(feat).lower(), int(coeffs), str(setup).lower(), int(fold), int(seed))
    return key in COMPLETED_SET


def execute_task3_run(dataset, feat_kind, n_coeff, setup, fold_idx, seed):
    keras.backend.clear_session()
    gc.collect()
    set_seed(seed)

    num_classes_map = {"emodb": 7, "cafe": 7, "aesdd": 5}
    num_classes = num_classes_map[dataset]

    run_id = f"{dataset}_{feat_kind}_{n_coeff}_fold{fold_idx}_{setup}_seed{seed}"
    model_ckpt_path = MODELS_DIR / f"ckpt_{run_id}.keras"
    temp_ckpt_path = Path(tempfile.gettempdir()) / f"ckpt_{run_id}.keras"

    start_time = time.time()

    # Data prep
    X_train, y_train, X_val, y_val, X_test, y_test, te_spk, va_spk = prepare_task3_fold_data(
        dataset, feat_kind, n_coeff, setup, fold_idx, seed
    )

    # Model architecture (identical to Week 3: lr=0.001, Adam, batch=32, epochs=100)
    input_shape = X_train.shape[1:]
    model = build_1d_cnn(
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=0.001,
        name=f"model_{run_id}",
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(temp_ckpt_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=0,
        )
    ]

    # Training (100 epochs, no early stopping, identical to Week 3)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
    )

    # Load best validation checkpoint
    best_model = keras.models.load_model(str(temp_ckpt_path))

    # Evaluate
    val_loss, val_acc = best_model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
    train_loss, train_acc = best_model.evaluate(X_train, y_train, verbose=0)
    elapsed = round(time.time() - start_time, 2)

    # Copy checkpoint and cleanup
    try:
        shutil.copy(str(temp_ckpt_path), str(model_ckpt_path))
    except Exception:
        pass
    try:
        temp_ckpt_path.unlink(missing_ok=True)
    except Exception:
        pass

    # Append to CSV
    row = {
        "dataset": dataset,
        "feature": feat_kind,
        "coeffs": n_coeff,
        "format": "baseline",
        "setup": setup,
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
        "duration_s": elapsed,
    }

    df_row = pd.DataFrame([row])
    if CSV_OUT.is_file():
        # Read existing to ensure all columns match
        existing_df = pd.read_csv(CSV_OUT)
        # Ensure new columns exist
        for col in ["fold", "test_spk", "val_spk"]:
            if col not in existing_df.columns:
                existing_df[col] = np.nan
        updated_df = pd.concat([existing_df, df_row], ignore_index=True)
        updated_df.to_csv(CSV_OUT, index=False)
    else:
        df_row.to_csv(CSV_OUT, index=False)

    COMPLETED_SET.add((str(dataset).lower(), str(feat_kind).lower(), int(n_coeff), str(setup).lower(), int(fold_idx), int(seed)))

    print(f"  [DONE] Fold {fold_idx} Seed {seed} ({setup}) in {elapsed}s | Test Acc: {test_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | Train Acc: {train_acc*100:.2f}%")
    return elapsed


def run_dataset_task3(dataset, n_folds):
    print("\n" + "="*70)
    print(f"STARTING TASK 3: {dataset.upper()} ({n_folds} Folds x 24 runs = {n_folds*24} runs)")
    print("="*70)

    ds_start = time.time()
    run_times = []
    completed_runs = 0

    for fold in range(n_folds):
        for feat in FEATURES:
            for c in COEFFS:
                for setup in SETUPS:
                    for s in SEEDS:
                        if is_already_run(dataset, feat, c, setup, fold, s):
                            print(f"[SKIP] Completed: {dataset}_{feat}_{c}_fold{fold}_{setup}_seed{s}")
                            continue
                        print(f"\n[RUN] {dataset}_{feat}_{c}_fold{fold}_{setup}_seed{s}")
                        t_elapsed = execute_task3_run(dataset, feat, c, setup, fold, s)
                        run_times.append(t_elapsed)
                        completed_runs += 1

    ds_total_time = round(time.time() - ds_start, 2)
    avg_run_time = round(np.mean(run_times), 2) if run_times else 0.0
    print(f"\n[SUMMARY] {dataset.upper()} finished {completed_runs} runs in {ds_total_time}s | Avg time per run: {avg_run_time}s")
    return avg_run_time, ds_total_time


def main():
    print("="*70)
    print("WEEK 4 TASK 3: ROTATING SPEAKER-DISJOINT FOLD EVALUATION (384 RUNS)")
    print("Order: EmoDB (5 folds) -> AESDD (5 folds) -> CAFE (6 folds)")
    print("Settings: 100 epochs, batch 32, lr 0.001, ModelCheckpoint on val_accuracy")
    print("="*70)

    refresh_completed_set()

    # Calculate missing runs
    expected_runs = []
    folds_map = {"emodb": 5, "aesdd": 5, "cafe": 6}
    for dataset, n_folds in folds_map.items():
        for fold in range(n_folds):
            for feat in FEATURES:
                for c in COEFFS:
                    for setup in SETUPS:
                        for s in SEEDS:
                            expected_runs.append((dataset, feat, c, setup, fold, s))

    missing_runs = [r for r in expected_runs if not is_already_run(*r)]
    print(f"\nTask 3 Total Expected: {len(expected_runs)} runs")
    print(f"Task 3 Completed: {len(COMPLETED_SET)} runs")
    print(f"Task 3 Missing to Run: {len(missing_runs)} runs")

    if missing_runs:
        print("\nDetailed List of Missing Runs to Execute:")
        for idx, (ds, f, c, st, fold, s) in enumerate(missing_runs):
            print(f"  {idx+1:2d}. {ds.upper()} | Fold {fold} | {f.upper()} {c} | {st} | Seed {s}")
    else:
        print("\nAll 384 Task 3 runs are already completed!")

    total_start = time.time()
    per_dataset_times = {}

    # 1. EmoDB
    t_avg_emodb, t_tot_emodb = run_dataset_task3("emodb", n_folds=5)
    per_dataset_times["emodb"] = {"avg_s": t_avg_emodb, "total_s": t_tot_emodb}

    # 2. AESDD
    t_avg_aesdd, t_tot_aesdd = run_dataset_task3("aesdd", n_folds=5)
    per_dataset_times["aesdd"] = {"avg_s": t_avg_aesdd, "total_s": t_tot_aesdd}

    # 3. CAFE
    t_avg_cafe, t_tot_cafe = run_dataset_task3("cafe", n_folds=6)
    per_dataset_times["cafe"] = {"avg_s": t_avg_cafe, "total_s": t_tot_cafe}

    grand_total_time = round(time.time() - total_start, 2)
    print("\n" + "="*70)
    print("ALL 384 TASK 3 EXPERIMENTS COMPLETE!")
    print(f"Grand Total Time: {grand_total_time/60:.2f} minutes")
    for ds, stats in per_dataset_times.items():
        print(f"  - {ds.upper()}: Avg {stats['avg_s']}s/run | Total {stats['total_s']/60:.2f} mins")
    
    # Confirm final row count is exactly 618
    final_df = pd.read_csv(CSV_OUT)
    print(f"\nFinal row count in {CSV_OUT.name}: {len(final_df)}")
    assert len(final_df) == 618, f"FATAL: Final row count is {len(final_df)}, expected 618!"
    print("Assertion PASSED: Final row count is exactly 618.")
    print("="*70)

    # Post-run aggregation
    print("\nRunning metric aggregation and plot generation...")
    import subprocess
    subprocess.run([sys.executable, str(ROOT_DIR / "scripts" / "aggregate_task3.py")], check=True)
    subprocess.run([sys.executable, str(ROOT_DIR / "scripts" / "generate_week4_summary.py")], check=True)
    subprocess.run([sys.executable, str(ROOT_DIR / "scripts" / "generate_week4_deck.py")], check=True)
    print("All post-run artifacts successfully generated!")


if __name__ == "__main__":
    main()
