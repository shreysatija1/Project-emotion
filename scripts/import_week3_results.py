"""
Import existing Week 3 EmoDB baseline runs into results/week4_results.csv.
Preserves existing Week 3 results without re-running them.
"""

import json
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
CSV_OUT = RESULTS_DIR / "week4_results.csv"

rows = []
for p in sorted(RESULTS_DIR.glob("summary_*.json")):
    with open(p, "r") as f:
        d = json.load(f)
    
    aug_mode = d.get("aug_mode")
    if aug_mode == "none":
        setup = "none"
    elif aug_mode == "after_split":
        setup = "version_a"
    elif aug_mode == "before_split":
        setup = "version_b"
    else:
        continue

    rows.append({
        "dataset": "emodb",
        "feature": d.get("feature_type"),
        "coeffs": d.get("n_coeff"),
        "format": "baseline",
        "setup": setup,
        "split_type": "stratified",
        "seed": d.get("seed"),
        "val_acc": round(float(d.get("val_accuracy", 0.0)), 4),
        "test_acc": round(float(d.get("test_accuracy", 0.0)), 4),
        "train_acc": round(float(d.get("train_accuracy", 0.0)), 4),
        "val_loss": round(float(d.get("val_loss", 0.0)), 4),
        "test_loss": round(float(d.get("test_loss", 0.0)), 4),
        "duration_s": 0.0,
    })

df = pd.DataFrame(rows)
if CSV_OUT.is_file():
    df_existing = pd.read_csv(CSV_OUT)
    df = pd.concat([df_existing, df], ignore_index=True).drop_duplicates(
        subset=["dataset", "feature", "coeffs", "format", "setup", "split_type", "seed"]
    )
df.to_csv(CSV_OUT, index=False)
print(f"Imported {len(rows)} Week 3 EmoDB baseline runs into {CSV_OUT.name}.")
