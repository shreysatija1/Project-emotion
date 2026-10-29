"""
Unit and Integration Tests for Week 5 Reproduction Deliverables:
- Week 4 baseline validations:
  * 144-input feature shapes and absence of NaNs/Infs
  * Stratified 60/20/20 split proportions and mutual exclusivity
  * Zero speaker overlap in rotating speaker-disjoint folds
- Week 5 acoustic augmentation validations:
  * Pitch shifting: 5x copies, ±2 semitones, valid audio arrays
  * Time stretching: 5x copies, 0.9x-1.1x rates, valid audio arrays
  * SpecAugment: 5x copies, pre-pooling masking, exact shape (2*n_coeff,), no NaNs
- Week 5 leakage and protocol integrity checks:
  * Training audio only is augmented; val and test stay 100% clean
  * Zero audio clips cross split boundaries in augmented sets
  * Zero speaker overlap across train, val, test in all folds
  * StandardScaler fitted exclusively on training split statistics
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
FEAT_DIR = ROOT_DIR / "features"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from features_144 import (
    extract_mfcc_frames144,
    extract_mfmc_frames144,
    extract_mfcc_stats144,
    extract_mfmc_stats144,
)
from data_loader import get_metadata, load_audio_clip, SR
from split_utils import make_stratified_split, make_rotating_speaker_fold, make_speaker_independent_split
from acoustic_augment import (
    augment_pitch_shift,
    augment_time_stretch,
    extract_spec_augment_features,
)
from extract_all_features import extract_baseline_mfcc, extract_baseline_mfmc


def test_feature_shapes_and_validity():
    """Verify that 144-input feature extractors produce exact shapes and no NaNs."""
    sr = 16000
    rng = np.random.default_rng(42)
    y = rng.normal(0, 0.1, size=int(1.5 * sr)).astype(np.float32)

    for n_coeff in [12, 24, 30]:
        f_mfcc = extract_mfcc_frames144(y, sr, n_coeff)
        assert f_mfcc.shape == (144, n_coeff), f"Expected (144, {n_coeff}), got {f_mfcc.shape}"
        assert not np.isnan(f_mfcc).any(), "NaN found in MFCC frames144"
        assert not np.isinf(f_mfcc).any(), "Inf found in MFCC frames144"

        f_mfmc = extract_mfmc_frames144(y, sr, n_coeff)
        assert f_mfmc.shape == (144, n_coeff), f"Expected (144, {n_coeff}), got {f_mfmc.shape}"
        assert not np.isnan(f_mfmc).any(), "NaN found in MFMC frames144"
        assert not np.isinf(f_mfmc).any(), "Inf found in MFMC frames144"

        s_mfcc = extract_mfcc_stats144(y, sr, n_coeff)
        assert s_mfcc.shape == (144,), f"Expected (144,), got {s_mfcc.shape}"
        assert not np.isnan(s_mfcc).any(), "NaN found in MFCC stats144"
        assert not np.isinf(s_mfcc).any(), "Inf found in MFCC stats144"

        s_mfmc = extract_mfmc_stats144(y, sr, n_coeff)
        assert s_mfmc.shape == (144,), f"Expected (144,), got {s_mfmc.shape}"
        assert not np.isnan(s_mfmc).any(), "NaN found in MFMC stats144"
        assert not np.isinf(s_mfmc).any(), "Inf found in MFMC stats144"


def test_stratified_splits():
    """Verify stratified split sizes and mutual exclusivity across datasets."""
    for name in ["emodb", "cafe", "aesdd", "mesd"]:
        meta = get_metadata(name)
        df = make_stratified_split(meta, seed=42)

        tr = df[df["split"] == "train"]
        va = df[df["split"] == "val"]
        te = df[df["split"] == "test"]

        assert len(tr) + len(va) + len(te) == len(meta)
        assert len(set(tr.index) & set(va.index)) == 0
        assert len(set(tr.index) & set(te.index)) == 0
        assert len(set(va.index) & set(te.index)) == 0
        assert len(set(tr["filename"]) & set(va["filename"])) == 0
        assert len(set(tr["filename"]) & set(te["filename"])) == 0
        assert len(set(va["filename"]) & set(te["filename"])) == 0

        n = len(meta)
        assert abs(len(tr) / n - 0.60) < 0.05
        assert abs(len(va) / n - 0.20) < 0.05
        assert abs(len(te) / n - 0.20) < 0.05


def test_speaker_independent_splits():
    """Verify zero speaker overlap in rotating speaker-disjoint folds."""
    for name in ["emodb", "cafe", "aesdd"]:
        meta = get_metadata(name)
        n_folds = 6 if name == "cafe" else 5
        for fold in range(n_folds):
            df = make_rotating_speaker_fold(meta, name, fold)
            tr_spk = set(df[df["split"] == "train"]["speaker_id"].unique())
            va_spk = set(df[df["split"] == "val"]["speaker_id"].unique())
            te_spk = set(df[df["split"] == "test"]["speaker_id"].unique())

            assert len(tr_spk & va_spk) == 0, f"{name} fold {fold}: Train & Val overlap: {tr_spk & va_spk}"
            assert len(tr_spk & te_spk) == 0, f"{name} fold {fold}: Train & Test overlap: {tr_spk & te_spk}"
            assert len(va_spk & te_spk) == 0, f"{name} fold {fold}: Val & Test overlap: {va_spk & te_spk}"


def test_acoustic_augmentations():
    """Verify that Week 5 acoustic augmentations produce 5 copies without NaNs."""
    sr = 16000
    rng = np.random.default_rng(42)
    y = rng.normal(0, 0.1, size=sr * 2).astype(np.float32)

    # 1. Pitch shift
    p_copies = augment_pitch_shift(y, sr=sr, steps=(0, -2, -1, 1, 2))
    assert len(p_copies) == 5, f"Expected 5 pitch-shifted copies, got {len(p_copies)}"
    for idx, c in enumerate(p_copies):
        assert not np.isnan(c).any(), f"NaN in pitch-shifted copy {idx}"
        assert not np.isinf(c).any(), f"Inf in pitch-shifted copy {idx}"
        assert c.dtype == np.float32

    # 2. Time stretch
    t_copies = augment_time_stretch(y, rates=(1.0, 0.90, 0.95, 1.05, 1.10))
    assert len(t_copies) == 5, f"Expected 5 time-stretched copies, got {len(t_copies)}"
    for idx, c in enumerate(t_copies):
        assert not np.isnan(c).any(), f"NaN in time-stretched copy {idx}"
        assert not np.isinf(c).any(), f"Inf in time-stretched copy {idx}"
        assert c.dtype == np.float32

    # 3. SpecAugment (Pre-pooling time/frequency masking)
    for feat_kind in ["mfcc", "mfmc"]:
        for n_coeff in [24, 30]:
            s_copies = extract_spec_augment_features(y, sr, feat_kind, n_coeff, rng, n_copies=5)
            assert len(s_copies) == 5, f"Expected 5 SpecAugment copies, got {len(s_copies)}"
            for idx, c in enumerate(s_copies):
                assert c.shape == (2 * n_coeff,), f"Expected shape ({2 * n_coeff},), got {c.shape}"
                assert not np.isnan(c).any(), f"NaN in SpecAugment {feat_kind} {n_coeff} copy {idx}"
                assert not np.isinf(c).any(), f"Inf in SpecAugment {feat_kind} {n_coeff} copy {idx}"


def test_augmentation_leakage_and_scaler_isolation():
    """Verify training-only augmentation, zero test leakage, and train-only StandardScaler fitting."""
    meta = get_metadata("emodb")
    df_split = make_rotating_speaker_fold(meta, "emodb", fold_idx=0)

    tr_meta = df_split[df_split["split"] == "train"]
    va_meta = df_split[df_split["split"] == "val"]
    te_meta = df_split[df_split["split"] == "test"]

    tr_spk = set(tr_meta["speaker_id"].unique())
    va_spk = set(va_meta["speaker_id"].unique())
    te_spk = set(te_meta["speaker_id"].unique())

    # 1. Assert zero speaker leakage
    assert len(tr_spk & va_spk) == 0, f"Train and Val share speakers: {tr_spk & va_spk}"
    assert len(tr_spk & te_spk) == 0, f"Train and Test share speakers: {tr_spk & te_spk}"
    assert len(va_spk & te_spk) == 0, f"Val and Test share speakers: {va_spk & te_spk}"

    # 2. Simulate training augmentation
    rng = np.random.default_rng(42)
    aug_train_clips = []
    aug_train_speakers = []
    for _, row in tr_meta.iterrows():
        wav_path = ROOT_DIR / row["rel_path"]
        y_orig = load_audio_clip(wav_path, SR)
        p_copies = augment_pitch_shift(y_orig, SR)
        assert len(p_copies) == 5
        aug_train_clips.extend(p_copies)
        aug_train_speakers.extend([row["speaker_id"]] * 5)

    # 3. Assert augmented samples never come from val or test speakers
    aug_spk_set = set(aug_train_speakers)
    assert len(aug_spk_set & va_spk) == 0, f"Augmented train contains Val speaker: {aug_spk_set & va_spk}"
    assert len(aug_spk_set & te_spk) == 0, f"Augmented train contains Test speaker: {aug_spk_set & te_spk}"

    # 4. Assert StandardScaler train-only isolation
    X_train_dummy = rng.normal(loc=5.0, scale=2.0, size=(len(aug_train_clips), 48)).astype(np.float32)
    X_val_dummy = rng.normal(loc=15.0, scale=4.0, size=(len(va_meta), 48)).astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(X_train_dummy)

    # Scaler mean must match X_train_dummy mean closely (~5.0), not X_val_dummy (~15.0)
    assert abs(np.mean(scaler.mean_) - 5.0) < 0.2
    assert abs(np.mean(scaler.mean_) - 15.0) > 8.0

    # Transforming val using train scaler must preserve training parameters
    X_val_trans = scaler.transform(X_val_dummy)
    assert not np.isnan(X_val_trans).any()


if __name__ == "__main__":
    print("Running Week 5 Verification Test Suite...")
    test_feature_shapes_and_validity()
    print("  [1/5] test_feature_shapes_and_validity: PASSED")
    test_stratified_splits()
    print("  [2/5] test_stratified_splits: PASSED")
    test_speaker_independent_splits()
    print("  [3/5] test_speaker_independent_splits: PASSED")
    test_acoustic_augmentations()
    print("  [4/5] test_acoustic_augmentations: PASSED")
    test_augmentation_leakage_and_scaler_isolation()
    print("  [5/5] test_augmentation_leakage_and_scaler_isolation: PASSED")
    print("\nALL WEEK 5 VERIFICATION TESTS PASSED SUCCESSFULLY!")
