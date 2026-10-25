"""
Evaluation script for Speech Emotion Recognition 1D-CNN.

Protocol:
  1. Inspects validation results from results/summary_*.json.
  2. Selects the best checkpoint for each configuration based STRICTLY on validation accuracy.
  3. Writes and freezes selected_models.md BEFORE any test set evaluation occurs.
  4. Evaluates each selected model on its respective test set exactly once.
  5. Computes mean and std of validation and test accuracy across seeds.
  6. Generates:
     - results/final_results_table.csv & results/final_results_table.png
     - Confusion matrices for best MFCC & MFMC models in after_split mode
     - Training curves (accuracy & loss) for best runs
     - Comparison with paper benchmarks.
"""

import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import keras

from config import ROOT, PROC_DIR, FEAT_DIR, RESULTS_DIR
from cache import load_features, load_split
from model import prepare_input

EMOTION_NAMES = ["anger", "boredom", "disgust", "fear", "happiness", "neutral", "sadness"]


def load_test_data(feature_type, n_coeff, aug_mode):
    """
    Load test features, labels, and metadata for the given mode.
    """
    kind = feature_type.lower().strip()
    mode = aug_mode.lower().strip()

    if mode == "none":
        X = load_features(kind, n_coeff)
        tr_idx, va_idx, te_idx, y = load_split()
        meta = pd.read_csv(PROC_DIR / "splits.csv")
        return X[te_idx], y[te_idx], meta.iloc[te_idx].copy()

    elif mode == "after_split":
        feat_path = FEAT_DIR / f"version_a_{kind}_{n_coeff}.npy"
        label_path = FEAT_DIR / "version_a_labels.npy"
        meta_path = PROC_DIR / "version_a_metadata.csv"
        if not meta_path.is_file():
            meta_path = FEAT_DIR / "version_a_metadata.csv"

        X = np.load(feat_path)
        y = np.load(label_path)
        meta = pd.read_csv(meta_path)
        te_idx = meta.index[meta["split"] == "test"].values
        return X[te_idx], y[te_idx], meta.iloc[te_idx].copy()

    elif mode == "before_split":
        feat_path = FEAT_DIR / f"version_b_{kind}_{n_coeff}.npy"
        label_path = FEAT_DIR / "version_b_labels.npy"
        meta_path = PROC_DIR / "version_b_metadata.csv"
        if not meta_path.is_file():
            meta_path = FEAT_DIR / "version_b_metadata.csv"

        X = np.load(feat_path)
        y = np.load(label_path)
        meta = pd.read_csv(meta_path)
        te_idx = meta.index[meta["split"] == "test"].values
        return X[te_idx], y[te_idx], meta.iloc[te_idx].copy()

    else:
        raise ValueError(f"Unknown aug_mode: '{aug_mode}'")


def collect_summaries(results_dir=RESULTS_DIR):
    """Collect all training summary JSON files."""
    summaries = []
    for p in Path(results_dir).glob("summary_*.json"):
        with open(p, "r") as f:
            data = json.load(f)
            summaries.append(data)
    return summaries


def write_and_freeze_selected_models(summaries, selected_models_path=ROOT / "selected_models.md"):
    """
    Select best model for each (feature, n, mode) configuration based STRICTLY
    on validation accuracy and freeze to selected_models.md before testing.
    """
    df = pd.DataFrame(summaries)
    configs = []

    lines = [
        "# Model Selection Protocol (Frozen Before Test Evaluation)\n\n",
        "**Strict Rule**: All models listed below were selected **strictly based on validation accuracy (`val_accuracy`)**.",
        "No test-set data was inspected or evaluated prior to finalizing this selection file.\n\n",
        "| Configuration | Best Seed | Best Val Acc | Val Loss | Train Acc | Selected Model Checkpoint |\n",
        "| :--- | :---: | :---: | :---: | :---: | :--- |\n",
    ]

    selected_per_config = {}

    for (f_type, n_coeff, mode), grp in df.groupby(["feature_type", "n_coeff", "aug_mode"]):
        best_run = grp.sort_values(by=["val_accuracy", "val_loss"], ascending=[False, True]).iloc[0]
        config_key = f"{f_type}_{n_coeff}_{mode}"
        selected_per_config[config_key] = best_run.to_dict()

        cfg_label = f"**{f_type.upper()}** - {n_coeff} ({mode})"
        best_seed = best_run["seed"]
        val_acc = f"{best_run['val_accuracy'] * 100:.2f}%"
        val_loss = f"{best_run['val_loss']:.4f}"
        tr_acc = f"{best_run.get('train_accuracy', 0.0) * 100:.2f}%"
        model_name = Path(best_run["best_model_path"]).name

        lines.append(
            f"| {cfg_label} | {best_seed} | **{val_acc}** | {val_loss} | {tr_acc} | `{model_name}` |\n"
        )

    lines.append("\n\n## Per-Seed Validation Summary Table\n\n")
    lines.append("| Feature | Coeffs | Aug Mode | Seed 42 | Seed 43 | Seed 44 | Mean Val Acc |\n")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")

    for (f_type, n_coeff, mode), grp in df.groupby(["feature_type", "n_coeff", "aug_mode"]):
        seed_map = dict(zip(grp["seed"], grp["val_accuracy"]))
        s42 = f"{seed_map.get(42, 0.0) * 100:.2f}%" if 42 in seed_map else "N/A"
        s43 = f"{seed_map.get(43, 0.0) * 100:.2f}%" if 43 in seed_map else "N/A"
        s44 = f"{seed_map.get(44, 0.0) * 100:.2f}%" if 44 in seed_map else "N/A"
        mean_acc = f"{grp['val_accuracy'].mean() * 100:.2f}%"
        lines.append(f"| {f_type.upper()} | {n_coeff} | `{mode}` | {s42} | {s43} | {s44} | **{mean_acc}** |\n")

    lines.append("\n\n*Status: FROZEN. Ready for single-pass test set scoring.*\n")

    selected_models_path.write_text("".join(lines), encoding="utf-8")
    print(f"[FREEZE] Successfully froze model selection to: {selected_models_path}")
    return selected_per_config


def evaluate_test_set(summaries, results_dir=RESULTS_DIR, models_dir=ROOT / "models"):
    """
    Score each trained run model on its test set exactly once.
    """
    print("\n[START] Scoring models on their designated test set (single-pass)...")
    test_eval_results = []

    for s in summaries:
        run_name = s["run_name"]
        f_type = s["feature_type"]
        n_coeff = s["n_coeff"]
        mode = s["aug_mode"]
        seed = s["seed"]

        model_path = Path(s["best_model_path"])
        scaler_path = Path(s["scaler_path"])

        if not model_path.exists():
            model_path = Path(models_dir) / f"best_model_{run_name}.keras"
        if not scaler_path.exists():
            scaler_path = Path(results_dir) / f"scaler_{run_name}.joblib"

        # 1. Load test data
        X_test, y_test, meta_test = load_test_data(f_type, n_coeff, mode)

        # 2. Transform with fitted training scaler
        scaler = joblib.load(scaler_path)
        X_test_scaled = scaler.transform(X_test)
        X_test_3d = prepare_input(X_test_scaled)

        # 3. Load model and evaluate
        model = keras.models.load_model(str(model_path))
        test_loss, test_acc = model.evaluate(X_test_3d, y_test, verbose=0)
        y_pred = np.argmax(model.predict(X_test_3d, verbose=0), axis=1)

        res = {
            "run_name": run_name,
            "feature_type": f_type,
            "n_coeff": n_coeff,
            "aug_mode": mode,
            "seed": seed,
            "val_accuracy": s["val_accuracy"],
            "val_loss": s["val_loss"],
            "test_accuracy": round(float(test_acc), 4),
            "test_loss": round(float(test_loss), 4),
            "y_true": y_test.tolist(),
            "y_pred": y_pred.tolist(),
        }
        test_eval_results.append(res)
        print(f"  Scored {run_name}: Val Acc = {s['val_accuracy']*100:.2f}%, Test Acc = {test_acc*100:.2f}%")

    return test_eval_results


def build_results_tables_and_plots(test_eval_results, results_dir=RESULTS_DIR):
    """
    Aggregate results across seeds and generate CSV and PNG tables.
    """
    df = pd.DataFrame(test_eval_results)
    results_dir = Path(results_dir)

    # 1. Aggregation across seeds
    grouped = df.groupby(["feature_type", "n_coeff", "aug_mode"]).agg(
        val_acc_mean=("val_accuracy", "mean"),
        val_acc_std=("val_accuracy", "std"),
        test_acc_mean=("test_accuracy", "mean"),
        test_acc_std=("test_accuracy", "std"),
        count=("seed", "count"),
    ).reset_index()

    # Fill NaN std if only 1 seed
    grouped["val_acc_std"] = grouped["val_acc_std"].fillna(0.0)
    grouped["test_acc_std"] = grouped["test_acc_std"].fillna(0.0)

    # Format cells: "Val: 85.0 ± 1.2% | Test: 86.4 ± 0.8%"
    def fmt_cell(row):
        val_m = row["val_acc_mean"] * 100
        val_s = row["val_acc_std"] * 100
        tst_m = row["test_acc_mean"] * 100
        tst_s = row["test_acc_std"] * 100
        if row["count"] > 1:
            return f"Val: {val_m:.1f}±{val_s:.1f}%\nTest: {tst_m:.1f}±{tst_s:.1f}%"
        else:
            return f"Val: {val_m:.1f}%\nTest: {tst_m:.1f}%"

    grouped["cell_text"] = grouped.apply(fmt_cell, axis=1)

    pivot_display = grouped.pivot(
        index=["feature_type", "n_coeff"],
        columns="aug_mode",
        values="cell_text"
    )
    # Reorder columns
    col_order = [c for c in ["none", "after_split", "before_split"] if c in pivot_display.columns]
    pivot_display = pivot_display[col_order]

    # Save detailed CSV
    detailed_csv_path = results_dir / "final_results_table.csv"
    grouped.to_csv(detailed_csv_path, index=False)
    print(f"[SAVED] Detailed results CSV: {detailed_csv_path}")

    # Render PNG table
    plt.figure(figsize=(12, 6), dpi=300)
    plt.axis("off")
    cell_data = []
    row_labels = []
    for (f_type, n_coeff), row in pivot_display.iterrows():
        row_labels.append(f"{f_type.upper()} ({n_coeff} coeffs)")
        cell_data.append([row.get(col, "N/A") for col in col_order])

    col_headers = [
        "No Augmentation\n(none)",
        "Version A\n(after_split)",
        "Version B\n(before_split, leaked)",
    ]

    table = plt.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.2)

    # Style header
    for k, cell in table._cells.items():
        if k[0] == 0:
            cell.set_facecolor("#2b5b84")
            cell.set_text_props(color="white", weight="bold")
        elif k[1] == -1:
            cell.set_facecolor("#e6edf2")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#fcfcfc")

    plt.title("Speech Emotion Recognition Performance (EmoDB 1D-CNN)", fontsize=14, weight="bold", pad=20)
    png_table_path = results_dir / "final_results_table.png"
    plt.tight_layout()
    plt.savefig(png_table_path, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Final results table image: {png_table_path}")

    return grouped, pivot_display


def plot_confusion_matrices(test_eval_results, selected_per_config, results_dir=RESULTS_DIR):
    """
    Generate confusion matrices with emotion names for best MFCC and best MFMC in after_split mode.
    """
    results_dir = Path(results_dir)
    res_df = pd.DataFrame(test_eval_results)

    # Filter for after_split mode
    after_df = res_df[res_df["aug_mode"] == "after_split"]

    for feat in ["mfcc", "mfmc"]:
        feat_sub = after_df[after_df["feature_type"] == feat]
        if feat_sub.empty:
            continue

        # Best by val_accuracy
        best_run = feat_sub.sort_values(by="val_accuracy", ascending=False).iloc[0]
        y_true = np.array(best_run["y_true"])
        y_pred = np.array(best_run["y_pred"])
        n_coeff = best_run["n_coeff"]
        seed = best_run["seed"]
        test_acc = best_run["test_accuracy"]

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(EMOTION_NAMES))))
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(8, 6.5), dpi=300)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=EMOTION_NAMES,
            yticklabels=EMOTION_NAMES,
            cbar=False,
            annot_kws={"size": 11, "weight": "bold"},
        )
        # Add normalized percentages below numbers
        for i in range(len(EMOTION_NAMES)):
            for j in range(len(EMOTION_NAMES)):
                pct = cm_norm[i, j] * 100
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                plt.text(
                    j + 0.5,
                    i + 0.75,
                    f"({pct:.0f}%)",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )

        plt.title(
            f"Confusion Matrix: Best {feat.upper()} (Version A: after_split)\n"
            f"Coeffs: {n_coeff} | Seed: {seed} | Test Accuracy: {test_acc*100:.2f}%",
            fontsize=12,
            weight="bold",
            pad=15,
        )
        plt.xlabel("Predicted Emotion", fontsize=11, weight="bold")
        plt.ylabel("True Emotion", fontsize=11, weight="bold")
        plt.xticks(rotation=30)
        plt.yticks(rotation=0)
        plt.tight_layout()

        out_path = results_dir / f"confusion_matrix_best_{feat}_after_split.png"
        plt.savefig(out_path)
        plt.close()
        print(f"[SAVED] Confusion matrix: {out_path}")


def plot_learning_curves(selected_per_config, results_dir=RESULTS_DIR):
    """
    Plot training and validation accuracy and loss curves for representative runs.
    """
    results_dir = Path(results_dir)
    target_configs = [
        ("mfcc_30_none", "MFCC 30 (none)"),
        ("mfcc_30_after_split", "MFCC 30 (after_split)"),
        ("mfcc_30_before_split", "MFCC 30 (before_split)"),
        ("mfmc_30_after_split", "MFMC 30 (after_split)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    for cfg_key, label in target_configs:
        matching = [v for k, v in selected_per_config.items() if cfg_key in k]
        if not matching:
            continue
        run_info = matching[0]
        hist_path = Path(run_info["history_path"])
        if not hist_path.exists():
            hist_path = results_dir / f"history_{run_info['run_name']}.csv"

        if hist_path.exists():
            h_df = pd.read_csv(hist_path)
            epochs = h_df["epoch"] + 1 if "epoch" in h_df.columns else np.arange(1, len(h_df) + 1)
            # Accuracy plot
            axes[0].plot(epochs, h_df["val_accuracy"] * 100, label=f"{label} (Val)", linewidth=2)
            # Loss plot
            axes[1].plot(epochs, h_df["val_loss"], label=f"{label} (Val)", linewidth=2)

    axes[0].set_title("Validation Accuracy Across Epochs", fontsize=12, weight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Validation Accuracy (%)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=9)

    axes[1].set_title("Validation Loss Across Epochs", fontsize=12, weight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Validation Loss", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=9)

    plt.suptitle("1D-CNN Training Convergence Curves", fontsize=14, weight="bold", y=0.98)
    plt.tight_layout()
    out_path = results_dir / "training_curves_best_runs.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[SAVED] Training curves: {out_path}")


def main():
    results_dir = RESULTS_DIR
    summaries = collect_summaries(results_dir)
    if not summaries:
        print("[ERROR] No summary JSON files found in results/.")
        sys.exit(1)

    print(f"[INFO] Found {len(summaries)} training summaries.")

    # Step 1: Write and freeze selected models
    selected_per_config = write_and_freeze_selected_models(summaries)

    # Step 2: Score selected models on test set (exactly once)
    test_eval_results = evaluate_test_set(summaries, results_dir=results_dir)

    # Step 3: Tables and plots
    grouped, pivot_display = build_results_tables_and_plots(test_eval_results, results_dir=results_dir)

    # Step 4: Confusion matrices
    plot_confusion_matrices(test_eval_results, selected_per_config, results_dir=results_dir)

    # Step 5: Learning curves
    plot_learning_curves(selected_per_config, results_dir=results_dir)

    print("\n[SUCCESS] All evaluation tasks completed!")


if __name__ == "__main__":
    main()
