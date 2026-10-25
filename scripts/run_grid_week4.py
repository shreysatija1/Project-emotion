"""
Grid Runner for Week 4 Experiments.

Supports resuming: automatically skips configurations already recorded in results/week4_results.csv.
Supports targeted batches via command-line arguments:
--task1 : Run EmoDB 144-input format experiments (frames144, stats144)
--task2 : Run multi-dataset stratified experiments on CAFE, AESDD, MESD
--task3 : Run speaker-independent split experiments on EmoDB, CAFE, AESDD
--all   : Run all tasks sequentially
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
RESULTS_DIR = ROOT_DIR / "results"
CSV_OUT = RESULTS_DIR / "week4_results.csv"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train_week4 import execute_run

SEEDS = [42, 43, 44]
COEFFS = [12, 24, 30]
FEATURES = ["mfcc", "mfmc"]


def is_already_run(dataset, feat, coeffs, fmt, setup, split_type, seed):
    """Check if the exact run has already been completed in week4_results.csv."""
    if not CSV_OUT.is_file():
        return False
    df = pd.read_csv(CSV_OUT)
    match = df[
        (df["dataset"] == dataset) &
        (df["feature"] == feat) &
        (df["coeffs"] == coeffs) &
        (df["format"] == fmt) &
        (df["setup"] == setup) &
        (df["split_type"] == split_type) &
        (df["seed"] == seed)
    ]
    return len(match) > 0


def run_task1_emodb_144():
    """Task 1: EmoDB 144-input investigations (frames144 and stats144)."""
    print("\n" + "="*70)
    print("STARTING TASK 1: EmoDB 144-Input Investigations (frames144 & stats144)")
    print("="*70)

    for fmt in ["frames144", "stats144"]:
        for feat in FEATURES:
            for c in COEFFS:
                for setup in ["none", "version_a"]:
                    for s in SEEDS:
                        if is_already_run("emodb", feat, c, fmt, setup, "stratified", s):
                            print(f"[SKIP] Already completed: emodb_{feat}_{c}_{fmt}_{setup}_stratified_seed{s}")
                            continue
                        execute_run(
                            dataset="emodb",
                            feat_kind=feat,
                            n_coeff=c,
                            feat_format=fmt,
                            setup=setup,
                            split_type="stratified",
                            seed=s,
                            epochs=100,
                            csv_out=CSV_OUT,
                        )


def run_task2_multidataset():
    """Task 2: Repeat baseline on CAFE, AESDD, MESD."""
    print("\n" + "="*70)
    print("STARTING TASK 2: Multi-Dataset Experiments (CAFE, AESDD, MESD)")
    print("="*70)

    for ds in ["cafe", "aesdd", "mesd"]:
        for feat in FEATURES:
            for c in COEFFS:
                for setup in ["none", "version_a"]:
                    for s in SEEDS:
                        if is_already_run(ds, feat, c, "baseline", setup, "stratified", s):
                            print(f"[SKIP] Already completed: {ds}_{feat}_{c}_baseline_{setup}_stratified_seed{s}")
                            continue
                        execute_run(
                            dataset=ds,
                            feat_kind=feat,
                            n_coeff=c,
                            feat_format="baseline",
                            setup=setup,
                            split_type="stratified",
                            seed=s,
                            epochs=100,
                            csv_out=CSV_OUT,
                        )


def run_task3_speaker_independent():
    """Task 3: Speaker-Independent Splits on EmoDB, CAFE, AESDD."""
    print("\n" + "="*70)
    print("STARTING TASK 3: Speaker-Independent Split Experiments")
    print("="*70)

    for ds in ["emodb", "cafe", "aesdd"]:
        for feat in FEATURES:
            for c in COEFFS:
                for setup in ["none", "version_a"]:
                    for s in SEEDS:
                        if is_already_run(ds, feat, c, "baseline", setup, "speaker_independent", s):
                            print(f"[SKIP] Already completed: {ds}_{feat}_{c}_baseline_{setup}_speaker_independent_seed{s}")
                            continue
                        execute_run(
                            dataset=ds,
                            feat_kind=feat,
                            n_coeff=c,
                            feat_format="baseline",
                            setup=setup,
                            split_type="speaker_independent",
                            seed=s,
                            epochs=100,
                            csv_out=CSV_OUT,
                        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task1", action="store_true", help="Run Task 1 (144-input formats on EmoDB)")
    parser.add_argument("--task2", action="store_true", help="Run Task 2 (CAFE, AESDD, MESD stratified)")
    parser.add_argument("--task3", action="store_true", help="Run Task 3 (Speaker-independent splits)")
    parser.add_argument("--all", action="store_true", help="Run all tasks")
    args = parser.parse_args()

    if args.task1 or args.all:
        run_task1_emodb_144()
    if args.task2 or args.all:
        run_task2_multidataset()
    if args.task3 or args.all:
        run_task3_speaker_independent()

    # Automatically generate summary tables, plots, and updated deck
    if args.task1 or args.task2 or args.task3 or args.all:
        print("\n" + "="*70)
        print("EXPERIMENT BATCH COMPLETE. GENERATING SUMMARY TABLES, PLOTS, AND DECK...")
        print("="*70)
        try:
            from generate_week4_summary import generate_all_summaries
            generate_all_summaries()
        except Exception as e:
            print(f"Summary generation notice: {e}")
        try:
            from generate_week4_deck import create_deck
            create_deck()
        except Exception as e:
            print(f"Deck generation notice: {e}")

    if not (args.task1 or args.task2 or args.task3 or args.all):
        print("Please specify --task1, --task2, --task3, or --all.")
