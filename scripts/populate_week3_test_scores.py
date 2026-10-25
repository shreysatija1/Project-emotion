"""
Populate exact test accuracy and test loss for Week 3 baseline models into week4_results.csv.
Evaluates the frozen Week 3 checkpoints on the exact Week 3 test split once.
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import keras

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"
CSV_FILE = RESULTS_DIR / "week4_results.csv"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import PROC_DIR, FEAT_DIR
from cache import load_features, load_split
from model import prepare_input


def update_week3_test_scores():
    if not CSV_FILE.is_file():
        print("week4_results.csv not found.")
        return

    df = pd.read_csv(CSV_FILE)
    
    # Load test split
    _, _, te_idx, y_all = load_split()
    y_test = y_all[te_idx]

    updated = 0
    for idx, row in df.iterrows():
        if row["dataset"] == "emodb" and row["format"] == "baseline" and row["test_acc"] == 0.0:
            feat = row["feature"]
            n = row["coeffs"]
            setup = row["setup"]
            seed = row["seed"]
            
            # Map setup back to Week 3 aug_mode
            aug_mode = "after_split" if setup == "version_a" else ("before_split" if setup == "version_b" else "none")
            run_name = f"{feat}_{n}_{aug_mode}_seed{seed}"
            model_path = MODELS_DIR / f"best_model_{run_name}.keras"
            scaler_path = RESULTS_DIR / f"scaler_{run_name}.joblib"
            
            if model_path.is_file() and scaler_path.is_file():
                X_all = load_features(feat, n)
                X_test = X_all[te_idx]
                scaler = joblib.load(scaler_path)
                X_test_scaled = scaler.transform(X_test)
                X_test_3d = prepare_input(X_test_scaled)
                
                model = keras.models.load_model(str(model_path))
                test_loss, test_acc = model.evaluate(X_test_3d, y_test, verbose=0)
                
                df.at[idx, "test_acc"] = round(float(test_acc), 4)
                df.at[idx, "test_loss"] = round(float(test_loss), 4)
                updated += 1
                del model
                keras.backend.clear_session()
                print(f"Updated {run_name}: test_acc={test_acc*100:.2f}%")

    if updated > 0:
        df.to_csv(CSV_FILE, index=False)
        print(f"Successfully updated {updated} Week 3 test scores in {CSV_FILE.name}.")
    else:
        print("No Week 3 rows needed test score updating.")


if __name__ == "__main__":
    update_week3_test_scores()
