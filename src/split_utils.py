"""
Splitting utilities:
1. Stratified 60/20/20 Split (by emotion label).
2. Speaker-Independent Split (strict disjointness across train, val, and test).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT_DIR / "data" / "processed"


def make_stratified_split(df, seed=42):
    """
    Create a stratified 60/20/20 split based on the 'label' column.
    Returns DataFrame with 'split' column containing 'train', 'val', 'test'.
    """
    df = df.copy()
    idx = df.index.values
    tr_idx, tmp_idx = train_test_split(
        idx, test_size=0.40, stratify=df["label"], random_state=seed
    )
    va_idx, te_idx = train_test_split(
        tmp_idx, test_size=0.50, stratify=df.loc[tmp_idx, "label"], random_state=seed
    )
    
    df["split"] = "train"
    df.loc[va_idx, "split"] = "val"
    df.loc[te_idx, "split"] = "test"
    
    # Sanity checks
    assert len(set(tr_idx) & set(va_idx)) == 0
    assert len(set(tr_idx) & set(te_idx)) == 0
    assert len(set(va_idx) & set(te_idx)) == 0
    return df


def make_speaker_independent_split(df, dataset_name, seed=42):
    """
    Create a speaker-independent split where train, val, and test sets
    have completely non-overlapping speaker IDs.
    
    Parameters
    ----------
    df : pd.DataFrame
        Metadata table containing 'speaker_id' and 'label'.
    dataset_name : str
        'emodb', 'cafe', 'aesdd', or 'mesd'.
    seed : int
        Random seed for speaker assignment where applicable.
        
    Returns
    -------
    pd.DataFrame with 'split' column ('train', 'val', 'test').
    """
    df = df.copy()
    name = dataset_name.lower().strip()
    rng = np.random.default_rng(seed)
    
    if name == "emodb":
        # 10 speakers: 5 male (03, 10, 11, 12, 15), 5 female (08, 09, 13, 14, 16)
        # Ensure gender balance in each split:
        # Male speakers
        m_speakers = sorted([s for s in df[df["gender"] == "m"]["speaker_id"].unique()])
        f_speakers = sorted([s for s in df[df["gender"] == "f"]["speaker_id"].unique()])
        
        m_perm = rng.permutation(m_speakers).tolist()
        f_perm = rng.permutation(f_speakers).tolist()
        
        # 3 male, 3 female to train (6 total)
        # 1 male, 1 female to val (2 total)
        # 1 male, 1 female to test (2 total)
        tr_spk = set(m_perm[:3] + f_perm[:3])
        va_spk = set([m_perm[3], f_perm[3]])
        te_spk = set([m_perm[4], f_perm[4]])

    elif name == "cafe":
        # 12 speakers: 6 male (odd numbers), 6 female (even numbers)
        m_speakers = sorted([s for s in df[df["gender"] == "m"]["speaker_id"].unique()])
        f_speakers = sorted([s for s in df[df["gender"] == "f"]["speaker_id"].unique()])
        
        m_perm = rng.permutation(m_speakers).tolist()
        f_perm = rng.permutation(f_speakers).tolist()
        
        # 4 male, 4 female to train (8 total, 66.7%)
        # 1 male, 1 female to val (2 total, 16.7%)
        # 1 male, 1 female to test (2 total, 16.7%)
        tr_spk = set(m_perm[:4] + f_perm[:4])
        va_spk = set([m_perm[4], f_perm[4]])
        te_spk = set([m_perm[5], f_perm[5]])

    elif name == "aesdd":
        # 5 actors (1, 2, 3, 4, 5) with 100 clips each
        actors = sorted(df["speaker_id"].unique().tolist())
        perm = rng.permutation(actors).tolist()
        
        # 3 actors to train (60%), 1 to val (20%), 1 to test (20%)
        tr_spk = set(perm[:3])
        va_spk = set([perm[3]])
        te_spk = set([perm[4]])

    elif name == "mesd":
        # MESD has 3 voice groups ('child', 'female', 'male')
        # Partitioning by group: female to train, child to val, male to test
        # (Documented as demographic cohort split)
        tr_spk = set(["female"])
        va_spk = set(["child"])
        te_spk = set(["male"])

    else:
        raise ValueError(f"Unknown dataset for speaker-independent split: {name}")

    # Assign split column
    df["split"] = "train"
    df.loc[df["speaker_id"].isin(va_spk), "split"] = "val"
    df.loc[df["speaker_id"].isin(te_spk), "split"] = "test"

    # STRICT ASSERTIONS: No speaker overlap
    actual_tr = set(df[df["split"] == "train"]["speaker_id"].unique())
    actual_va = set(df[df["split"] == "val"]["speaker_id"].unique())
    actual_te = set(df[df["split"] == "test"]["speaker_id"].unique())

    assert len(actual_tr & actual_va) == 0, f"Leakage detected between train and val: {actual_tr & actual_va}"
    assert len(actual_tr & actual_te) == 0, f"Leakage detected between train and test: {actual_tr & actual_te}"
    assert len(actual_va & actual_te) == 0, f"Leakage detected between val and test: {actual_va & actual_te}"

    return df


if __name__ == "__main__":
    from data_loader import get_metadata
    for name in ["emodb", "cafe", "aesdd", "mesd"]:
        meta = get_metadata(name)
        df_strat = make_stratified_split(meta, seed=42)
        print(f"=== {name.upper()} STRATIFIED ===")
        print(df_strat["split"].value_counts())
        
        df_spk = make_speaker_independent_split(meta, name, seed=42)
        print(f"=== {name.upper()} SPEAKER-INDEPENDENT ===")
        print(df_spk["split"].value_counts())
        print(df_spk.groupby(["split", "speaker_id"]).size())
        print()
