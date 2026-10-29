"""
Phase 0: Dataset Inspection, Unified 6-Class Label Mapping, and Protocol Splitting (P1 and P2).

Protocols:
- P1: Speaker-grouped cross-validation over pooled (EmoDB + CAFE + AESDD).
      Aim for 5-6 folds with 2-3 speakers per fold balanced across corpora.
- P2: Leave-One-Corpus-Out (test on EmoDB, test on CAFE, test on AESDD).
- MESD: Mexican Spanish clips used strictly in training splits (never in val or test).
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
WEEK6_DIR = ROOT_DIR / "week6_webmodel"
DATA_DIR = ROOT_DIR / "data" / "processed"

TARGET_EMOTIONS = ["anger", "disgust", "fear", "happiness", "neutral", "sadness"]
EMO2IDX = {e: i for i, e in enumerate(TARGET_EMOTIONS)}
IDX2EMO = {i: e for i, e in enumerate(TARGET_EMOTIONS)}


def build_unified_dataset():
    """
    Load EmoDB, CAFE, AESDD, and MESD metadata, filter to the 6 target emotions,
    and create unique speaker identifiers prefixed with the corpus name.
    """
    dfs = []
    
    # 1. EmoDB
    emodb_df = pd.read_csv(DATA_DIR / "emodb_metadata.csv")
    emodb_df["corpus"] = "emodb"
    # drop 'boredom'
    emodb_df = emodb_df[emodb_df["emotion"].isin(TARGET_EMOTIONS)].copy()
    emodb_df["full_speaker_id"] = "emodb_" + emodb_df["speaker_id"].astype(str).str.zfill(2)
    dfs.append(emodb_df)
    
    # 2. CAFE
    cafe_df = pd.read_csv(DATA_DIR / "cafe_metadata.csv")
    cafe_df["corpus"] = "cafe"
    # drop 'surprise'
    cafe_df = cafe_df[cafe_df["emotion"].isin(TARGET_EMOTIONS)].copy()
    cafe_df["full_speaker_id"] = cafe_df["speaker_id"].astype(str)
    dfs.append(cafe_df)
    
    # 3. AESDD
    aesdd_df = pd.read_csv(DATA_DIR / "aesdd_metadata.csv")
    aesdd_df["corpus"] = "aesdd"
    # AESDD has anger, disgust, fear, happiness, sadness (no neutral)
    aesdd_df = aesdd_df[aesdd_df["emotion"].isin(TARGET_EMOTIONS)].copy()
    aesdd_df["full_speaker_id"] = "aesdd_" + aesdd_df["speaker_id"].astype(str)
    dfs.append(aesdd_df)
    
    # 4. MESD (for train sets only)
    mesd_df = pd.read_csv(DATA_DIR / "mesd_metadata.csv")
    mesd_df["corpus"] = "mesd"
    mesd_df = mesd_df[mesd_df["emotion"].isin(TARGET_EMOTIONS)].copy()
    mesd_df["full_speaker_id"] = "mesd_" + mesd_df["speaker_id"].astype(str)
    dfs.append(mesd_df)
    
    all_df = pd.concat(dfs, ignore_index=True)
    all_df["target_label"] = all_df["emotion"].map(EMO2IDX)
    
    # Resolve absolute audio paths
    all_df["audio_path"] = all_df["rel_path"].apply(lambda p: str((ROOT_DIR / p).resolve()))
    
    return all_df


def create_p1_speaker_folds(pooled_df, n_folds=5, seed=42):
    """
    Primary Protocol P1: Speaker-grouped cross validation over EmoDB, CAFE, AESDD.
    Each fold holds out balanced speakers from each corpus:
    - EmoDB has 10 speakers (5 groups of 2: 1M, 1F) -> 2 test speakers, 2 val speakers, 6 train speakers per fold.
    - CAFE has 12 speakers (6 groups of 2: 1M, 1F) -> 2 test speakers, 2 val speakers, 8 train speakers per fold.
    - AESDD has 5 core actors (1-5), actor 6 in train -> 1 test actor, 1 val actor, 3 train actors per fold.
    
    This guarantees 5 comprehensive folds where every fold has:
    - 2 EmoDB + 2 CAFE + 1 AESDD test speakers = 5 test speakers (~250-350 test clips)
    - 2 EmoDB + 2 CAFE + 1 AESDD val speakers = 5 val speakers
    - Remainder train speakers
    Zero leakage: Train, Val, and Test NEVER share any speaker.
    """
    folds = []
    
    # EmoDB: 5 pairs
    emodb_spk_pairs = [
        ["emodb_03", "emodb_08"],
        ["emodb_10", "emodb_09"],
        ["emodb_11", "emodb_13"],
        ["emodb_12", "emodb_14"],
        ["emodb_15", "emodb_16"]
    ]
    
    # CAFE: 6 pairs (use 5 pairs for 5 folds, pair 5 always in train)
    cafe_spk_pairs = [
        ["cafe_01", "cafe_02"],
        ["cafe_03", "cafe_04"],
        ["cafe_05", "cafe_06"],
        ["cafe_07", "cafe_08"],
        ["cafe_09", "cafe_10"],
        ["cafe_11", "cafe_12"]
    ]
    
    # AESDD: 5 core actors
    aesdd_core = ["aesdd_1", "aesdd_2", "aesdd_3", "aesdd_4", "aesdd_5"]
    
    for fold_idx in range(n_folds):
        # EmoDB test/val
        te_emodb = set(emodb_spk_pairs[fold_idx % 5])
        va_emodb = set(emodb_spk_pairs[(fold_idx + 1) % 5])
        tr_emodb = set([s for p in emodb_spk_pairs for s in p if s not in te_emodb and s not in va_emodb])
        
        # CAFE test/val (5 folds rotation over 6 pairs)
        te_cafe = set(cafe_spk_pairs[fold_idx % 6])
        va_cafe = set(cafe_spk_pairs[(fold_idx + 1) % 6])
        tr_cafe = set([s for p in cafe_spk_pairs for s in p if s not in te_cafe and s not in va_cafe])
        
        # AESDD test/val (5 core actors rotation)
        te_aesdd = set([aesdd_core[fold_idx % 5]])
        va_aesdd = set([aesdd_core[(fold_idx + 1) % 5]])
        tr_aesdd = set([s for s in aesdd_core if s not in te_aesdd and s not in va_aesdd] + ["aesdd_6"])
        
        test_spks = te_emodb | te_cafe | te_aesdd
        val_spks = va_emodb | va_cafe | va_aesdd
        train_spks = tr_emodb | tr_cafe | tr_aesdd
        
        # Verify strict disjointness
        assert len(test_spks & val_spks) == 0
        assert len(test_spks & train_spks) == 0
        assert len(val_spks & train_spks) == 0
        
        fold_dict = {
            "fold": fold_idx,
            "train_speakers": sorted(list(train_spks)),
            "val_speakers": sorted(list(val_spks)),
            "test_speakers": sorted(list(test_spks))
        }
        folds.append(fold_dict)
        
    return folds


def create_p2_loco_folds(pooled_df):
    """
    Secondary Protocol P2: Leave-One-Corpus-Out (LOCO).
    Folds:
    1. Test = EmoDB. Val = CAFE 2 speakers + AESDD 1 speaker. Train = Remainder CAFE + AESDD.
    2. Test = CAFE. Val = EmoDB 2 speakers + AESDD 1 speaker. Train = Remainder EmoDB + AESDD.
    3. Test = AESDD. Val = EmoDB 2 speakers + CAFE 2 speakers. Train = Remainder EmoDB + CAFE.
    """
    loco_folds = []
    
    # Fold 1: Test = EmoDB
    loco_folds.append({
        "fold_name": "loco_emodb",
        "test_corpus": "emodb",
        "val_speakers": ["cafe_01", "cafe_02", "aesdd_1"],
        "test_corpus_all": True
    })
    
    # Fold 2: Test = CAFE
    loco_folds.append({
        "fold_name": "loco_cafe",
        "test_corpus": "cafe",
        "val_speakers": ["emodb_03", "emodb_08", "aesdd_2"],
        "test_corpus_all": True
    })
    
    # Fold 3: Test = AESDD
    loco_folds.append({
        "fold_name": "loco_aesdd",
        "test_corpus": "aesdd",
        "val_speakers": ["emodb_10", "emodb_09", "cafe_03", "cafe_04"],
        "test_corpus_all": True
    })
    
    return loco_folds


if __name__ == "__main__":
    df = build_unified_dataset()
    WEEK6_DIR.mkdir(parents=True, exist_ok=True)
    (WEEK6_DIR / "results").mkdir(exist_ok=True)
    (WEEK6_DIR / "models").mkdir(exist_ok=True)
    (WEEK6_DIR / "cache").mkdir(exist_ok=True)
    
    df.to_csv(WEEK6_DIR / "unified_metadata.csv", index=False)
    
    print("=== UNIFIED DATASET SUMMARY ===")
    print(f"Total rows: {len(df)}")
    print(f"Corpora counts:\n{df['corpus'].value_counts()}")
    print(f"\nTarget Emotion counts across all corpora:\n{df['emotion'].value_counts()}")
    print(f"\nCorpus x Emotion breakdown:\n{pd.crosstab(df['corpus'], df['emotion'])}")
    
    # Check class imbalance
    counts = df[df['corpus'].isin(['emodb', 'cafe', 'aesdd'])]['emotion'].value_counts()
    imbalance_ratio = counts.max() / counts.min()
    print(f"\nPooled (EmoDB + CAFE + AESDD) class counts:\n{counts}")
    print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")
    
    p1_folds = create_p1_speaker_folds(df)
    print(f"\nGenerated {len(p1_folds)} P1 speaker-disjoint folds successfully.")
    for f in p1_folds:
        print(f"Fold {f['fold']}: Train {len(f['train_speakers'])} spks, Val {len(f['val_speakers'])} spks, Test {len(f['test_speakers'])} spks")
