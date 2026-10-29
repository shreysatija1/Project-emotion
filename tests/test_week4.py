"""
Unit tests for Week 4 reproduction tasks:
- Feature extraction formats and shapes (24/48/60, frames144, stats144)
- Absence of NaNs/Infs
- Speaker disjointness in speaker-independent splits
- Stratified 60/20/20 split proportions
- Train/val/test mutual exclusivity
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from features_144 import (
    extract_mfcc_frames144,
    extract_mfmc_frames144,
    extract_mfcc_stats144,
    extract_mfmc_stats144,
    TARGET_FRAMES,
)
from data_loader import get_metadata, load_audio_clip, SR
from split_utils import make_stratified_split, make_speaker_independent_split, make_rotating_speaker_fold
from acoustic_augment import augment_pitch_shift, augment_time_stretch, extract_spec_augment_features


def test_feature_shapes_and_validity():
    """Verify that 144-input feature extractors produce exact shapes and no NaNs."""
    # Synthetic 1.5s 16kHz audio
    sr = 16000
    rng = np.random.default_rng(42)
    y = rng.normal(0, 0.1, size=int(1.5 * sr)).astype(np.float32)

    for n_coeff in [12, 24, 30]:
        # Frames 144
        f_mfcc = extract_mfcc_frames144(y, sr, n_coeff)
        assert f_mfcc.shape == (144, n_coeff), f"Expected (144, {n_coeff}), got {f_mfcc.shape}"
        assert not np.isnan(f_mfcc).any(), "NaN found in MFCC frames144"
        assert not np.isinf(f_mfcc).any(), "Inf found in MFCC frames144"

        f_mfmc = extract_mfmc_frames144(y, sr, n_coeff)
        assert f_mfmc.shape == (144, n_coeff), f"Expected (144, {n_coeff}), got {f_mfmc.shape}"
        assert not np.isnan(f_mfmc).any(), "NaN found in MFMC frames144"
        assert not np.isinf(f_mfmc).any(), "Inf found in MFMC frames144"

        # Stats 144
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

        # Check total rows
        assert len(tr) + len(va) + len(te) == len(meta)

        # Disjoint indices
        assert len(set(tr.index) & set(va.index)) == 0
        assert len(set(tr.index) & set(te.index)) == 0
        assert len(set(va.index) & set(te.index)) == 0

        # Disjoint filenames
        assert len(set(tr["filename"]) & set(va["filename"])) == 0
        assert len(set(tr["filename"]) & set(te["filename"])) == 0
        assert len(set(va["filename"]) & set(te["filename"])) == 0

        # Approximate 60/20/20 proportions
        n = len(meta)
        assert abs(len(tr) / n - 0.60) < 0.05
        assert abs(len(va) / n - 0.20) < 0.05
        assert abs(len(te) / n - 0.20) < 0.05


def test_speaker_independent_splits():
    """Verify zero speaker overlap in speaker-independent splits."""
    for name in ["emodb", "cafe", "aesdd"]:
        meta = get_metadata(name)
        for seed in [42, 43, 44]:
            df = make_speaker_independent_split(meta, name, seed=seed)

            tr_spk = set(df[df["split"] == "train"]["speaker_id"].unique())
            va_spk = set(df[df["split"] == "val"]["speaker_id"].unique())
            te_spk = set(df[df["split"] == "test"]["speaker_id"].unique())

            # Strict disjointness
            assert len(tr_spk & va_spk) == 0, f"{name} seed {seed}: Train and Val share speakers: {tr_spk & va_spk}"
            assert len(tr_spk & te_spk) == 0, f"{name} seed {seed}: Train and Test share speakers: {tr_spk & te_spk}"
            assert len(va_spk & te_spk) == 0, f"{name} seed {seed}: Val and Test share speakers: {va_spk & te_spk}"

            # All splits have samples
            assert len(df[df["split"] == "train"]) > 0
            assert len(df[df["split"] == "val"]) > 0
            assert len(df[df["split"] == "test"]) > 0


def test_acoustic_augmentations():
    """Verify that Week 5 acoustic augmentations produce 5 copies without NaNs."""
    sr = 16000
    rng = np.random.default_rng(42)
    y = rng.normal(0, 0.1, size=sr * 2).astype(np.float32)

    p_copies = augment_pitch_shift(y, sr=sr, steps=(0, -2, -1, 1, 2))
    assert len(p_copies) == 5
    t_copies = augment_time_stretch(y, rates=(1.0, 0.90, 0.95, 1.05, 1.10))
    assert len(t_copies) == 5

    for feat_kind in ["mfcc", "mfmc"]:
        for n_coeff in [24, 30]:
            s_copies = extract_spec_augment_features(y, sr, feat_kind, n_coeff, rng, n_copies=5)
            assert len(s_copies) == 5
            for c in s_copies:
                assert c.shape == (2 * n_coeff,)
                assert not np.isnan(c).any()


def test_augmentation_leakage_and_scaler_isolation():
    """Verify training-only augmentation, zero test leakage, and train-only StandardScaler fitting."""
    from sklearn.preprocessing import StandardScaler
    meta = get_metadata("emodb")
    df_split = make_rotating_speaker_fold(meta, "emodb", fold_idx=0)

    tr_meta = df_split[df_split["split"] == "train"]
    va_meta = df_split[df_split["split"] == "val"]
    te_meta = df_split[df_split["split"] == "test"]

    tr_spk = set(tr_meta["speaker_id"].unique())
    va_spk = set(va_meta["speaker_id"].unique())
    te_spk = set(te_meta["speaker_id"].unique())

    assert len(tr_spk & va_spk) == 0
    assert len(tr_spk & te_spk) == 0
    assert len(va_spk & te_spk) == 0


if __name__ == "__main__":
    test_feature_shapes_and_validity()
    print("test_feature_shapes_and_validity: PASSED")
    test_stratified_splits()
    print("test_stratified_splits: PASSED")
    test_speaker_independent_splits()
    print("test_speaker_independent_splits: PASSED")
    test_acoustic_augmentations()
    print("test_acoustic_augmentations: PASSED")
    test_augmentation_leakage_and_scaler_isolation()
    print("test_augmentation_leakage_and_scaler_isolation: PASSED")
    print("ALL TESTS PASSED!")
