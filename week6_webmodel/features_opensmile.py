"""
C1: openSMILE eGeMAPSv02 feature extraction.
Extracts the standard 88-dimensional eGeMAPS acoustic feature set for every clip.
Per-utterance deployable extraction with zero corpus leakage.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import opensmile

ROOT_DIR = Path(__file__).resolve().parents[1]
WEEK6_DIR = ROOT_DIR / "week6_webmodel"
CACHE_DIR = WEEK6_DIR / "cache"

from data_manager import get_unified_metadata, load_clip, SR


def extract_egemaps_features(audio_path, smile_instance=None):
    """Extract 88-dimensional eGeMAPSv02 feature vector from an audio file."""
    if smile_instance is None:
        smile_instance = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    try:
        df_feat = smile_instance.process_file(str(audio_path))
        vec = df_feat.iloc[0].values.astype(np.float32)
    except Exception as e:
        # Fallback if processing file fails: load clip and process signal
        y = load_clip(audio_path, target_sr=SR)
        df_feat = smile_instance.process_signal(y, SR)
        vec = df_feat.iloc[0].values.astype(np.float32)
        
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec


def get_cached_egemaps_features(df):
    """Extract or load cached 88-dim eGeMAPS features for all clips in df."""
    cache_file = CACHE_DIR / "egemaps_88_features.npy"
    if cache_file.exists():
        return np.load(cache_file)
        
    print(f"[C1 eGeMAPS] Extracting eGeMAPSv02 (88-dim) features for {len(df)} clips...")
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    features = []
    for idx, row in df.iterrows():
        feat = extract_egemaps_features(row["audio_path"], smile_instance=smile)
        features.append(feat)
        if (idx + 1) % 500 == 0 or (idx + 1) == len(df):
            print(f"  Processed {idx + 1}/{len(df)} clips...")
            
    X = np.vstack(features).astype(np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, X)
    print(f"[C1 eGeMAPS] Saved cached features to {cache_file} (shape: {X.shape})")
    return X


def _worker_extract_librosa(audio_path):
    from features_handcrafted import extract_handcrafted_features
    y = load_clip(audio_path, target_sr=SR)
    return extract_handcrafted_features(y, sr=SR)


def get_cached_librosa_features(df, max_workers=8):
    """Extract or load cached 442-dim Librosa handcrafted features for all clips in df."""
    cache_file = CACHE_DIR / "librosa_442_features.npy"
    if cache_file.exists():
        return np.load(cache_file)
        
    from concurrent.futures import ProcessPoolExecutor
    print(f"[C1 Librosa] Extracting comprehensive handcrafted features (442-dim) for {len(df)} clips using {max_workers} workers...")
    
    paths = df["audio_path"].tolist()
    features = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for idx, feat in enumerate(executor.map(_worker_extract_librosa, paths, chunksize=16)):
            features.append(feat)
            if (idx + 1) % 500 == 0 or (idx + 1) == len(df):
                print(f"  Processed {idx + 1}/{len(df)} clips...")
                
    X = np.vstack(features).astype(np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, X)
    print(f"[C1 Librosa] Saved cached features to {cache_file} (shape: {X.shape})")
    return X


if __name__ == "__main__":
    df = get_unified_metadata()
    X_ege = get_cached_egemaps_features(df)
    print("eGeMAPS features shape:", X_ege.shape)
