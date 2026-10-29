"""
Data Manager for Week 6 Web Model.
Handles unified metadata, speaker-disjoint folds (P1 & P2), audio loading, and feature caching.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
WEEK6_DIR = ROOT_DIR / "week6_webmodel"
CACHE_DIR = WEEK6_DIR / "cache"

TARGET_EMOTIONS = ["anger", "disgust", "fear", "happiness", "neutral", "sadness"]
EMO2IDX = {e: i for i, e in enumerate(TARGET_EMOTIONS)}
IDX2EMO = {i: e for i, e in enumerate(TARGET_EMOTIONS)}

SR = 16000


def get_unified_metadata():
    """Load or build the unified 6-class metadata DataFrame."""
    meta_path = WEEK6_DIR / "unified_metadata.csv"
    if not meta_path.exists():
        from week6_webmodel.setup_phase0 import build_unified_dataset
        df = build_unified_dataset()
        df.to_csv(meta_path, index=False)
    else:
        df = pd.read_csv(meta_path)
    return df


def load_clip(audio_path, target_sr=SR, max_duration=10.0):
    """
    Load an audio clip, convert to mono, resample to target_sr,
    cap length at max_duration seconds, and peak-normalize.
    """
    try:
        y, orig_sr = sf.read(str(audio_path), dtype="float32")
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if orig_sr != target_sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr, res_type="soxr_mq")
    except Exception:
        y, _ = librosa.load(str(audio_path), sr=target_sr, mono=True)
        
    # Cap duration at max_duration seconds
    max_samples = int(max_duration * target_sr)
    if len(y) > max_samples:
        y = y[:max_samples]
        
    # Peak normalization per clip (deployability constraint)
    peak = np.max(np.abs(y)) if len(y) > 0 else 0.0
    if peak > 1e-6:
        y = y / peak
        
    return y.astype(np.float32)


def get_p1_folds():
    """Return the 5 standard P1 speaker-disjoint folds."""
    from week6_webmodel.setup_phase0 import create_p1_speaker_folds
    df = get_unified_metadata()
    return create_p1_speaker_folds(df, n_folds=5, seed=42)


def get_p2_folds():
    """Return the 3 standard P2 LOCO folds."""
    from week6_webmodel.setup_phase0 import create_p2_loco_folds
    df = get_unified_metadata()
    return create_p2_loco_folds(df)


def get_split_indices_for_p1(df, fold_info, include_mesd_in_train=False):
    """
    Given unified metadata df and fold_info dict:
    - Train: clips with full_speaker_id in train_speakers (+ MESD if requested)
    - Val: clips with full_speaker_id in val_speakers (never MESD)
    - Test: clips with full_speaker_id in test_speakers (never MESD)
    Returns train_idx, val_idx, test_idx as boolean arrays or index lists.
    """
    tr_spks = set(fold_info["train_speakers"])
    va_spks = set(fold_info["val_speakers"])
    te_spks = set(fold_info["test_speakers"])
    
    # Assert zero speaker overlap
    assert len(tr_spks & va_spks) == 0
    assert len(tr_spks & te_spks) == 0
    assert len(va_spks & te_spks) == 0
    
    is_eval_pool = df["corpus"].isin(["emodb", "cafe", "aesdd"])
    
    val_mask = is_eval_pool & df["full_speaker_id"].isin(va_spks)
    test_mask = is_eval_pool & df["full_speaker_id"].isin(te_spks)
    
    if include_mesd_in_train:
        train_mask = (is_eval_pool & df["full_speaker_id"].isin(tr_spks)) | (df["corpus"] == "mesd")
    else:
        train_mask = is_eval_pool & df["full_speaker_id"].isin(tr_spks)
        
    # Final assertion: test and val sets NEVER contain MESD
    assert df.loc[val_mask, "corpus"].isin(["emodb", "cafe", "aesdd"]).all()
    assert df.loc[test_mask, "corpus"].isin(["emodb", "cafe", "aesdd"]).all()
    
    # Assert zero overlap between index sets
    tr_indices = set(df[train_mask].index)
    va_indices = set(df[val_mask].index)
    te_indices = set(df[test_mask].index)
    
    assert len(tr_indices & va_indices) == 0
    assert len(tr_indices & te_indices) == 0
    assert len(va_indices & te_indices) == 0
    
    return np.where(train_mask)[0], np.where(val_mask)[0], np.where(test_mask)[0]


def get_split_indices_for_p2(df, loco_info, include_mesd_in_train=False):
    """
    Given unified metadata df and loco_info dict:
    - Test: all clips of test_corpus
    - Val: specified val_speakers from the remaining corpora
    - Train: all remaining clips of other corpora (+ MESD if requested)
    """
    test_corpus = loco_info["test_corpus"]
    va_spks = set(loco_info["val_speakers"])
    
    test_mask = (df["corpus"] == test_corpus)
    val_mask = (df["corpus"] != test_corpus) & (df["corpus"].isin(["emodb", "cafe", "aesdd"])) & df["full_speaker_id"].isin(va_spks)
    
    if include_mesd_in_train:
        train_mask = (df["corpus"] != test_corpus) & (~df["full_speaker_id"].isin(va_spks))
    else:
        train_mask = (df["corpus"] != test_corpus) & (df["corpus"].isin(["emodb", "cafe", "aesdd"])) & (~df["full_speaker_id"].isin(va_spks))
        
    # Assertions
    assert df.loc[val_mask, "corpus"].ne(test_corpus).all()
    assert not test_mask.any() or not val_mask.any() or len(set(df[test_mask].index) & set(df[val_mask].index)) == 0
    assert len(set(df[train_mask].index) & set(df[val_mask].index)) == 0
    assert len(set(df[train_mask].index) & set(df[test_mask].index)) == 0
    
    return np.where(train_mask)[0], np.where(val_mask)[0], np.where(test_mask)[0]
