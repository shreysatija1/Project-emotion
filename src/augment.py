"""
Data augmentation and feature extraction for EmoDB Speech Emotion Recognition.

Paper: "Data augmentation using a 1D-CNN model with MFCC/MFMC features for speech emotion recognition"
       Automatika (2024).

Implements:
  - 5 augmented sets per clip:
      set1: original waveform (noise level = 0.0)
      set2: zero-noise copy of original (noise level = 0.0)
      set3: set2 + Gaussian noise 0.01
      set4: set2 + Gaussian noise 0.025
      set5: original + Gaussian noise 0.05
    where noise is level * rng.standard_normal(len(wave)) with rng = np.random.default_rng(42).
  - Version A: Augment train clips only (5 sets each), keep val/test as original (set1).
  - Version B: Augment all clips (5 sets each), then perform a new stratified 60/20/20 split.
  - Feature extraction: MFCC and MFMC with 12, 24, and 30 coefficients.
  - Artifact saving: .npy feature matrices, .npy labels, and metadata CSVs.
  - Automated tests:
      1. No source clip crosses splits in Version A.
      2. No NaN or infinite values in features, labels, or metadata.
      3. Correct dimensions and shapes across all versions and coefficient configurations.
      4. Features for set2 (zero-noise copy) equal features for set1 (original).
"""

import sys
from pathlib import Path

# Ensure project root and src directory are in sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import ROOT, RAW_DIR, PROC_DIR, FEAT_DIR, SR, COEFF_LIST, SEED
from data import load_audio
from features_mfcc import extract_mfcc
from features_mfmc import extract_mfmc

# In-memory cache for loaded audio waveforms
_AUDIO_CACHE = {}


def extract_features(wave, sr=SR, feature_type="mfcc", n_coeff=30):
    """
    Extract MFCC or MFMC features pooled over time (mean + std).

    Parameters
    ----------
    wave : np.ndarray
        1D audio time series waveform.
    sr : int
        Sample rate in Hz (default: 16000).
    feature_type : str
        Feature type: 'mfcc' or 'mfmc'.
    n_coeff : int
        Number of coefficients (12, 24, or 30).

    Returns
    -------
    np.ndarray
        1D feature vector of shape (2 * n_coeff,) containing concatenated mean and std.
    """
    ft = str(feature_type).lower().strip()
    if ft == "mfcc":
        return extract_mfcc(wave, sr=sr, n_coeff=n_coeff)
    elif ft == "mfmc":
        return extract_mfmc(wave, sr=sr, n_coeff=n_coeff)
    else:
        raise ValueError(f"Unknown feature_type: '{feature_type}'. Expected 'mfcc' or 'mfmc'.")


def resolve_audio_path(clip_info, audio_dir=RAW_DIR):
    """
    Resolve the absolute path to an audio file from a Series, dict, or clip ID.

    Parameters
    ----------
    clip_info : pd.Series, dict, or str
        Row metadata or clip identifier.
    audio_dir : Path or str
        Directory containing raw audio files.

    Returns
    -------
    Path
        Path to the resolved audio file.
    """
    audio_dir = Path(audio_dir)
    if isinstance(clip_info, (pd.Series, dict)):
        if "filename" in clip_info and pd.notna(clip_info["filename"]):
            fn = str(clip_info["filename"])
            p = audio_dir / fn
            if p.is_file():
                return p
        if "clip_id" in clip_info and pd.notna(clip_info["clip_id"]):
            cid = str(clip_info["clip_id"])
            p = audio_dir / cid
            if p.is_file():
                return p
            p_wav = audio_dir / f"{cid}.wav"
            if p_wav.is_file():
                return p_wav
    else:
        cid = str(clip_info)
        p = audio_dir / cid
        if p.is_file():
            return p
        p_wav = audio_dir / f"{cid}.wav"
        if p_wav.is_file():
            return p_wav

    val = clip_info.get("clip_id", clip_info) if isinstance(clip_info, (pd.Series, dict)) else clip_info
    stem = Path(str(val)).stem
    matches = list(audio_dir.glob(f"{stem}.wav"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Cannot resolve audio file for '{clip_info}' in '{audio_dir}'.")


def get_audio(clip_info, audio_dir=RAW_DIR, cache=_AUDIO_CACHE):
    """
    Load an audio waveform with in-memory caching to avoid repeated disk reads.
    """
    path = resolve_audio_path(clip_info, audio_dir=audio_dir)
    path_key = str(path.resolve())
    if cache is not None and path_key in cache:
        return cache[path_key]
    wave = load_audio(path)
    if cache is not None:
        cache[path_key] = wave
    return wave


def build_augmented_sets(wave, rng=None, seed=SEED):
    """
    Build 5 sets per clip:
      - set1: original waveform
      - set2: zero-noise copy
      - set3: set2 + Gaussian noise 0.01
      - set4: set2 + 0.025
      - set5: original + 0.05
    where noise is level * rng.standard_normal(len(wave)) with rng = np.random.default_rng(42).

    Parameters
    ----------
    wave : np.ndarray
        1D audio time series waveform.
    rng : np.random.Generator, optional
        NumPy random number generator. If None, initialized with seed (default 42).
    seed : int
        RNG seed if rng is not provided.

    Returns
    -------
    list of dict
        List containing dicts with keys: 'set_name', 'noise_level', 'wave'.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    orig = np.asarray(wave)
    dtype = orig.dtype if orig.dtype in (np.float32, np.float64) else np.float32

    set1 = orig.copy()
    set2 = orig.copy()

    noise3 = 0.01 * rng.standard_normal(len(orig), dtype=dtype)
    set3 = set2 + noise3

    noise4 = 0.025 * rng.standard_normal(len(orig), dtype=dtype)
    set4 = set2 + noise4

    noise5 = 0.05 * rng.standard_normal(len(orig), dtype=dtype)
    set5 = orig + noise5

    return [
        {"set_name": "set1", "noise_level": 0.0, "wave": set1},
        {"set_name": "set2", "noise_level": 0.0, "wave": set2},
        {"set_name": "set3", "noise_level": 0.01, "wave": set3},
        {"set_name": "set4", "noise_level": 0.025, "wave": set4},
        {"set_name": "set5", "noise_level": 0.05, "wave": set5},
    ]


def load_split_csv(split_input=None):
    """
    Load split CSV or validate an existing DataFrame.
    Guarantees 'clip_id', 'split', and 'label' columns exist.
    """
    if split_input is None:
        split_input = PROC_DIR / "splits.csv"
    if isinstance(split_input, (str, Path)):
        df = pd.read_csv(split_input)
    elif isinstance(split_input, pd.DataFrame):
        df = split_input.copy()
    else:
        raise ValueError(f"Unsupported split input type: {type(split_input)}")

    if "clip_id" not in df.columns:
        if "filename" in df.columns:
            df["clip_id"] = df["filename"].apply(lambda x: Path(x).stem)
        else:
            df["clip_id"] = [f"clip_{i:04d}" for i in range(len(df))]

    if "split" not in df.columns:
        raise ValueError("Input split CSV must contain a 'split' column.")
    if "label" not in df.columns:
        raise ValueError("Input split CSV must contain a 'label' column.")

    return df


def augment_version_a(
    split_df=None,
    audio_dir=RAW_DIR,
    rng=None,
    seed=SEED,
    feature_extractor=extract_features,
    coeff_list=COEFF_LIST,
    sr=SR
):
    """
    Version A: Augment train clips only, keep val/test as original.

    - Train clips (321 clips) -> 5 sets each (1,605 clips, split='train').
    - Val clips (107 clips) -> set1 only (107 clips, split='val', noise_level=0.0).
    - Test clips (107 clips) -> set1 only (107 clips, split='test', noise_level=0.0).
    Total: 1,819 clips.

    Returns
    -------
    features_dict : dict
        Dict mapping '{kind}_{n}' (e.g. 'mfcc_12', 'mfmc_30') to 2D np.ndarray.
    labels : np.ndarray
        1D integer emotion labels.
    meta_df : pd.DataFrame
        DataFrame with columns: clip_id, source_clip_id, set_name, noise_level, label, split.
    """
    split_df = load_split_csv(split_df)
    if rng is None:
        rng = np.random.default_rng(seed)

    records = []
    waves = []

    for _, row in split_df.iterrows():
        src_id = str(row["clip_id"])
        base_id = Path(src_id).stem
        label = int(row["label"])
        split = str(row["split"]).lower().strip()
        wave = get_audio(row, audio_dir=audio_dir)

        if split == "train":
            sets = build_augmented_sets(wave, rng=rng)
            for s in sets:
                records.append({
                    "clip_id": f"{base_id}_{s['set_name']}",
                    "source_clip_id": src_id,
                    "set_name": s["set_name"],
                    "noise_level": float(s["noise_level"]),
                    "label": label,
                    "split": "train",
                })
                waves.append(s["wave"])
        elif split in ("val", "test"):
            records.append({
                "clip_id": f"{base_id}_set1",
                "source_clip_id": src_id,
                "set_name": "set1",
                "noise_level": 0.0,
                "label": label,
                "split": split,
            })
            waves.append(wave.copy())
        else:
            raise ValueError(f"Unrecognized split: '{split}' in row: {row}")

    meta_df = pd.DataFrame(records)
    labels = meta_df["label"].to_numpy(dtype=np.int64)

    features_dict = {}
    for n in coeff_list:
        features_dict[f"mfcc_{n}"] = np.zeros((len(waves), 2 * n), dtype=np.float32)
        features_dict[f"mfmc_{n}"] = np.zeros((len(waves), 2 * n), dtype=np.float32)

    for i, w in enumerate(tqdm(waves, desc="Extracting features (Version A)")):
        for n in coeff_list:
            features_dict[f"mfcc_{n}"][i] = feature_extractor(w, sr=sr, feature_type="mfcc", n_coeff=n)
            features_dict[f"mfmc_{n}"][i] = feature_extractor(w, sr=sr, feature_type="mfmc", n_coeff=n)

    return features_dict, labels, meta_df


def augment_version_b(
    split_df=None,
    audio_dir=RAW_DIR,
    rng=None,
    seed=SEED,
    feature_extractor=extract_features,
    coeff_list=COEFF_LIST,
    sr=SR
):
    """
    Version B: Augment all clips, then do a new stratified 60/20/20 split.

    - All clips (535 clips) -> 5 sets each (2,675 clips).
    - New stratified 60/20/20 split across all 2,675 clips:
        train: 1,605 clips (60%)
        val:   535 clips (20%)
        test:  535 clips (20%)

    Returns
    -------
    features_dict : dict
        Dict mapping '{kind}_{n}' (e.g. 'mfcc_12', 'mfmc_30') to 2D np.ndarray.
    labels : np.ndarray
        1D integer emotion labels.
    meta_df : pd.DataFrame
        DataFrame with columns: clip_id, source_clip_id, set_name, noise_level, label, split.
    """
    split_df = load_split_csv(split_df)
    if rng is None:
        rng = np.random.default_rng(seed)

    records = []
    waves = []

    for _, row in split_df.iterrows():
        src_id = str(row["clip_id"])
        base_id = Path(src_id).stem
        label = int(row["label"])
        wave = get_audio(row, audio_dir=audio_dir)

        sets = build_augmented_sets(wave, rng=rng)
        for s in sets:
            records.append({
                "clip_id": f"{base_id}_{s['set_name']}",
                "source_clip_id": src_id,
                "set_name": s["set_name"],
                "noise_level": float(s["noise_level"]),
                "label": label,
                "split": "unassigned",
            })
            waves.append(s["wave"])

    meta_df = pd.DataFrame(records)
    n_samples = len(meta_df)

    # Perform stratified 60/20/20 split
    indices = np.arange(n_samples)
    tr_idx, tmp_idx = train_test_split(
        indices,
        test_size=0.4,
        stratify=meta_df["label"],
        random_state=seed
    )
    val_idx, te_idx = train_test_split(
        tmp_idx,
        test_size=0.5,
        stratify=meta_df.loc[tmp_idx, "label"],
        random_state=seed
    )

    meta_df["split"] = "train"
    meta_df.loc[val_idx, "split"] = "val"
    meta_df.loc[te_idx, "split"] = "test"

    labels = meta_df["label"].to_numpy(dtype=np.int64)

    features_dict = {}
    for n in coeff_list:
        features_dict[f"mfcc_{n}"] = np.zeros((len(waves), 2 * n), dtype=np.float32)
        features_dict[f"mfmc_{n}"] = np.zeros((len(waves), 2 * n), dtype=np.float32)

    for i, w in enumerate(tqdm(waves, desc="Extracting features (Version B)")):
        for n in coeff_list:
            features_dict[f"mfcc_{n}"][i] = feature_extractor(w, sr=sr, feature_type="mfcc", n_coeff=n)
            features_dict[f"mfmc_{n}"][i] = feature_extractor(w, sr=sr, feature_type="mfmc", n_coeff=n)

    return features_dict, labels, meta_df


def save_augmented_data(
    version_name,
    features_dict,
    labels,
    metadata_df,
    feat_dir=FEAT_DIR,
    proc_dir=PROC_DIR
):
    """
    Save .npy features, labels, and metadata CSV for a given augmentation version.
    """
    feat_dir = Path(feat_dir)
    proc_dir = Path(proc_dir)
    feat_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    v = version_name.lower().strip()
    if not v.startswith("version_"):
        v = f"version_{v}"

    # Save feature matrices
    for key, arr in features_dict.items():
        np.save(feat_dir / f"{v}_{key}.npy", arr)
        np.save(feat_dir / f"emodb_{v}_{key}.npy", arr)

    # Save labels
    np.save(feat_dir / f"{v}_labels.npy", labels)
    np.save(feat_dir / f"emodb_{v}_labels.npy", labels)

    # Save metadata CSV
    cols = ["clip_id", "source_clip_id", "set_name", "noise_level", "label", "split"]
    meta_to_save = metadata_df[cols]
    meta_to_save.to_csv(feat_dir / f"{v}_metadata.csv", index=False)
    meta_to_save.to_csv(proc_dir / f"{v}_metadata.csv", index=False)
    meta_to_save.to_csv(feat_dir / f"emodb_{v}_metadata.csv", index=False)
    meta_to_save.to_csv(proc_dir / f"emodb_{v}_metadata.csv", index=False)

    print(f"Successfully saved {v} artifacts to {feat_dir} and {proc_dir}")


def test_no_source_clip_crosses_splits_version_a(meta_a):
    """Check that no source clip crosses train, val, or test splits in Version A."""
    train_sources = set(meta_a.loc[meta_a["split"] == "train", "source_clip_id"])
    val_sources = set(meta_a.loc[meta_a["split"] == "val", "source_clip_id"])
    test_sources = set(meta_a.loc[meta_a["split"] == "test", "source_clip_id"])

    overlap_train_val = train_sources & val_sources
    overlap_train_test = train_sources & test_sources
    overlap_val_test = val_sources & test_sources

    assert len(overlap_train_val) == 0, f"Split leakage: train and val share source clips: {overlap_train_val}"
    assert len(overlap_train_test) == 0, f"Split leakage: train and test share source clips: {overlap_train_test}"
    assert len(overlap_val_test) == 0, f"Split leakage: val and test share source clips: {overlap_val_test}"
    assert len(train_sources) == 321, f"Expected 321 train source clips, got {len(train_sources)}"
    assert len(val_sources) == 107, f"Expected 107 val source clips, got {len(val_sources)}"
    assert len(test_sources) == 107, f"Expected 107 test source clips, got {len(test_sources)}"
    print("[PASS] Test: No source clip crosses splits in Version A.")


def test_no_nans(feats_a, labels_a, meta_a, feats_b, labels_b, meta_b):
    """Check that there are no NaN or infinite values in features, labels, or metadata."""
    for version_name, feats, labels, meta in [
        ("Version A", feats_a, labels_a, meta_a),
        ("Version B", feats_b, labels_b, meta_b),
    ]:
        for k, v in feats.items():
            assert not np.isnan(v).any(), f"NaN detected in {version_name} feature {k}"
            assert np.isfinite(v).all(), f"Non-finite value detected in {version_name} feature {k}"
        assert not np.isnan(labels).any(), f"NaN detected in {version_name} labels"
        assert not meta.isna().any().any(), f"NaN detected in {version_name} metadata"

    print("[PASS] Test: No NaNs or infinite values detected across Version A and B.")


def test_correct_shapes(meta_a, feats_a, labels_a, meta_b, feats_b, labels_b):
    """Check that feature arrays, labels, and metadata have the expected shapes."""
    # Version A: 321 train * 5 + 107 val * 1 + 107 test * 1 = 1819 rows
    expected_rows_a = 1819
    assert len(meta_a) == expected_rows_a, f"Expected {expected_rows_a} rows in meta_a, got {len(meta_a)}"
    assert len(labels_a) == expected_rows_a, f"Expected {expected_rows_a} labels_a, got {len(labels_a)}"
    for n in COEFF_LIST:
        expected_feat_dim = 2 * n
        assert feats_a[f"mfcc_{n}"].shape == (expected_rows_a, expected_feat_dim), (
            f"Shape mismatch for Version A mfcc_{n}: expected ({expected_rows_a}, {expected_feat_dim}), "
            f"got {feats_a[f'mfcc_{n}'].shape}"
        )
        assert feats_a[f"mfmc_{n}"].shape == (expected_rows_a, expected_feat_dim), (
            f"Shape mismatch for Version A mfmc_{n}: expected ({expected_rows_a}, {expected_feat_dim}), "
            f"got {feats_a[f'mfmc_{n}'].shape}"
        )

    # Version B: 535 total * 5 = 2675 rows
    expected_rows_b = 2675
    assert len(meta_b) == expected_rows_b, f"Expected {expected_rows_b} rows in meta_b, got {len(meta_b)}"
    assert len(labels_b) == expected_rows_b, f"Expected {expected_rows_b} labels_b, got {len(labels_b)}"
    for n in COEFF_LIST:
        expected_feat_dim = 2 * n
        assert feats_b[f"mfcc_{n}"].shape == (expected_rows_b, expected_feat_dim), (
            f"Shape mismatch for Version B mfcc_{n}: expected ({expected_rows_b}, {expected_feat_dim}), "
            f"got {feats_b[f'mfcc_{n}'].shape}"
        )
        assert feats_b[f"mfmc_{n}"].shape == (expected_rows_b, expected_feat_dim), (
            f"Shape mismatch for Version B mfmc_{n}: expected ({expected_rows_b}, {expected_feat_dim}), "
            f"got {feats_b[f'mfmc_{n}'].shape}"
        )

    print("[PASS] Test: Correct shapes confirmed for all feature matrices, labels, and metadata.")


def test_set2_features_equal_set1_features(meta_df, feats_dict, name="Dataset"):
    """Check that set2 (zero-noise copy) features equal set1 (original) features."""
    s1_df = meta_df[meta_df["set_name"] == "set1"]
    s2_df = meta_df[meta_df["set_name"] == "set2"]
    common_source_ids = sorted(list(set(s1_df["source_clip_id"]) & set(s2_df["source_clip_id"])))
    assert len(common_source_ids) > 0, "No common source clips with both set1 and set2 found"

    s1_map = {row["source_clip_id"]: idx for idx, row in s1_df.iterrows()}
    s2_map = {row["source_clip_id"]: idx for idx, row in s2_df.iterrows()}

    idx1 = np.array([s1_map[cid] for cid in common_source_ids])
    idx2 = np.array([s2_map[cid] for cid in common_source_ids])

    for feat_name, X in feats_dict.items():
        diff = np.max(np.abs(X[idx1] - X[idx2]))
        assert np.allclose(X[idx1], X[idx2], atol=1e-5), (
            f"Features for set1 and set2 do not match for {feat_name} in {name}: max diff = {diff}"
        )

    print(f"[PASS] Test: Set2 features equal Set1 features across all {len(common_source_ids)} pairs in {name}.")


def run_all_tests(meta_a, feats_a, labels_a, meta_b, feats_b, labels_b):
    """Run the complete test suite verifying all project constraints."""
    print("\n=================== RUNNING VERIFICATION TESTS ===================")
    test_no_source_clip_crosses_splits_version_a(meta_a)
    test_no_nans(feats_a, labels_a, meta_a, feats_b, labels_b, meta_b)
    test_correct_shapes(meta_a, feats_a, labels_a, meta_b, feats_b, labels_b)
    test_set2_features_equal_set1_features(meta_a, feats_a, name="Version A")
    test_set2_features_equal_set1_features(meta_b, feats_b, name="Version B")
    print("=================== ALL VERIFICATION TESTS PASSED ===================\n")


def load_saved_augmented_data(version_name, feat_dir=FEAT_DIR, proc_dir=PROC_DIR):
    """
    Load saved .npy features, labels, and metadata CSV for Version A or Version B.
    """
    feat_dir = Path(feat_dir)
    proc_dir = Path(proc_dir)
    v = version_name.lower().strip()
    if not v.startswith("version_"):
        v = f"version_{v}"

    meta_path = proc_dir / f"{v}_metadata.csv"
    if not meta_path.is_file():
        meta_path = feat_dir / f"{v}_metadata.csv"
    meta_df = pd.read_csv(meta_path)

    label_path = feat_dir / f"{v}_labels.npy"
    labels = np.load(label_path)

    features_dict = {}
    for n in COEFF_LIST:
        features_dict[f"mfcc_{n}"] = np.load(feat_dir / f"{v}_mfcc_{n}.npy")
        features_dict[f"mfmc_{n}"] = np.load(feat_dir / f"{v}_mfmc_{n}.npy")

    return features_dict, labels, meta_df


def load_augmented_split(version_name="version_a", proc_dir=PROC_DIR):
    """
    Load train, val, and test split indices along with labels for Version A or Version B.

    Parameters
    ----------
    version_name : str
        'version_a' or 'version_b'.
    proc_dir : Path or str
        Directory containing metadata CSV files (default: PROC_DIR).

    Returns
    -------
    train_idx : np.ndarray
    val_idx : np.ndarray
    test_idx : np.ndarray
    labels : np.ndarray
    """
    v = version_name.lower().strip()
    if not v.startswith("version_"):
        v = f"version_{v}"
    meta_path = Path(proc_dir) / f"{v}_metadata.csv"
    if not meta_path.is_file():
        meta_path = Path(FEAT_DIR) / f"{v}_metadata.csv"
    df = pd.read_csv(meta_path)
    return (
        df.index[df["split"] == "train"].values,
        df.index[df["split"] == "val"].values,
        df.index[df["split"] == "test"].values,
        df["label"].to_numpy(dtype=np.int64),
    )


def test_saved_artifacts(feat_dir=FEAT_DIR, proc_dir=PROC_DIR):
    """
    Load saved artifacts from disk and run all verification tests.
    """
    feats_a, labels_a, meta_a = load_saved_augmented_data("version_a", feat_dir, proc_dir)
    feats_b, labels_b, meta_b = load_saved_augmented_data("version_b", feat_dir, proc_dir)
    run_all_tests(meta_a, feats_a, labels_a, meta_b, feats_b, labels_b)


def main():
    if "--test-only" in sys.argv or "--test" in sys.argv:
        print("===== Running Augmentation Verification Tests on Saved Artifacts =====")
        test_saved_artifacts()
        return

    print("===== Running Augmentation & Feature Extraction Pipeline =====")

    # 1. Augment Version A
    print("\n--- Generating Version A (augment train clips only) ---")
    feats_a, labels_a, meta_a = augment_version_a()
    save_augmented_data("version_a", feats_a, labels_a, meta_a)
    print(f"Version A complete: {len(meta_a)} samples.")
    print("Split distribution:")
    print(meta_a["split"].value_counts().to_string())

    # 2. Augment Version B
    print("\n--- Generating Version B (augment all clips, stratified 60/20/20 split) ---")
    feats_b, labels_b, meta_b = augment_version_b()
    save_augmented_data("version_b", feats_b, labels_b, meta_b)
    print(f"Version B complete: {len(meta_b)} samples.")
    print("Split distribution:")
    print(meta_b["split"].value_counts().to_string())

    # 3. Run all tests
    run_all_tests(meta_a, feats_a, labels_a, meta_b, feats_b, labels_b)


if __name__ == "__main__":
    main()
