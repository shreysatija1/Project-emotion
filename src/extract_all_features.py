"""
Feature extraction and caching for all datasets and feature representations.
Supports:
- baseline: mean + std over time (shape: 2 * n_coeff)
- frames144: sequence of 144 frames (shape: 144, n_coeff)
- stats144: 144-dimensional extended summary vector (shape: 144)
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
FEAT_DIR = ROOT_DIR / "features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import get_metadata, load_audio_clip, SR
from features_144 import (
    extract_raw_mfcc, extract_raw_mfmc,
    extract_mfcc_frames144, extract_mfmc_frames144,
    extract_mfcc_stats144, extract_mfmc_stats144
)

COEFF_LIST = [12, 24, 30]


def extract_baseline_mfcc(y, sr, n):
    raw = extract_raw_mfcc(y, sr, n)
    return np.concatenate([raw.mean(axis=1), raw.std(axis=1)]).astype(np.float32)


def extract_baseline_mfmc(y, sr, n):
    raw = extract_raw_mfmc(y, sr, n)
    return np.concatenate([raw.mean(axis=1), raw.std(axis=1)]).astype(np.float32)


def extract_and_cache_dataset_features(dataset_name, formats=("baseline", "frames144", "stats144")):
    """Extract and save features for all clips in dataset_name."""
    name = dataset_name.lower().strip()
    meta = get_metadata(name)
    n_clips = len(meta)
    print(f"\n[CACHE FEATURES] Dataset: {name.upper()} ({n_clips} clips)")

    # Preload audio into memory or load on the fly
    print(f"Loading audio waveforms for {name}...")
    audio_list = []
    for _, row in tqdm(meta.iterrows(), total=n_clips, desc=f"Loading {name} audio"):
        wav_path = ROOT_DIR / row["rel_path"]
        y = load_audio_clip(wav_path, SR)
        audio_list.append(y)

    for fmt in formats:
        for feat_kind in ["mfcc", "mfmc"]:
            for n in COEFF_LIST:
                cache_file = FEAT_DIR / f"{name}_{feat_kind}_{n}_{fmt}.npy"
                if cache_file.is_file():
                    print(f"  Already cached: {cache_file.name}")
                    continue

                print(f"  Extracting {name} {feat_kind.upper()}-{n} ({fmt})...")
                feats = []
                for y in audio_list:
                    if fmt == "baseline":
                        f = extract_baseline_mfcc(y, SR, n) if feat_kind == "mfcc" else extract_baseline_mfmc(y, SR, n)
                    elif fmt == "frames144":
                        f = extract_mfcc_frames144(y, SR, n) if feat_kind == "mfcc" else extract_mfmc_frames144(y, SR, n)
                    elif fmt == "stats144":
                        f = extract_mfcc_stats144(y, SR, n) if feat_kind == "mfcc" else extract_mfmc_stats144(y, SR, n)
                    else:
                        raise ValueError(f"Unknown format: {fmt}")
                    feats.append(f)

                arr = np.array(feats, dtype=np.float32)
                np.save(cache_file, arr)
                print(f"    Saved: {cache_file.name} (shape: {arr.shape})")

    # Save canonical labels array
    labels_file = FEAT_DIR / f"{name}_labels.npy"
    if not labels_file.is_file():
        labels = meta["label"].values.astype(np.int32)
        np.save(labels_file, labels)
        print(f"  Saved labels: {labels_file.name} (shape: {labels.shape})")


if __name__ == "__main__":
    for d in ["emodb", "cafe", "aesdd", "mesd"]:
        extract_and_cache_dataset_features(d)
    print("\nALL DATASET FEATURES CACHED SUCCESSFULLY!")
