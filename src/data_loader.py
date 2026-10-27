"""
Unified metadata builder and data loader for all 4 speech emotion databases:
- EmoDB (German, 535 clips, 7 emotions)
- CAFE (Canadian French, 936 clips, 7 emotions)
- AESDD (Greek, 500 clips, 5 emotions)
- MESD (Mexican Spanish, 862 clips, 6 emotions)
"""

import os
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

SR = 16000

# Canonical Emotion Maps per Dataset
EMOTION_MAPS = {
    "emodb": {
        "W": "anger", "L": "boredom", "E": "disgust", "A": "fear",
        "F": "happiness", "T": "sadness", "N": "neutral"
    },
    "cafe": {
        "anger": "anger", "disgust": "disgust", "fear": "fear",
        "happiness": "happiness", "neutral": "neutral", "sadness": "sadness", "surprise": "surprise"
    },
    "aesdd": {
        "anger": "anger", "disgust": "disgust", "fear": "fear",
        "happiness": "happiness", "sadness": "sadness"
    },
    "mesd": {
        "anger": "anger", "disgust": "disgust", "fear": "fear",
        "happiness": "happiness", "neutral": "neutral", "sadness": "sadness"
    }
}

EMODB_SPEAKER_GENDER = {
    "03": "m", "08": "f", "09": "f", "10": "m", "11": "m",
    "12": "m", "13": "f", "14": "f", "15": "m", "16": "f"
}


def load_audio_clip(path, target_sr=SR):
    """Load an audio file resampled to target_sr mono float32."""
    y, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return y


def build_emodb_metadata():
    emodb_dir = RAW_DIR / "emodb"
    files = sorted(emodb_dir.glob("*.wav"))
    rows = []
    for f in files:
        stem = f.stem
        spk = stem[:2]
        code = stem[5]
        emo = EMOTION_MAPS["emodb"][code]
        gender = EMODB_SPEAKER_GENDER[spk]
        # Duration from audio
        dur = round(len(load_audio_clip(f)) / SR, 3)
        rows.append({
            "dataset": "emodb",
            "clip_id": stem,
            "filename": f.name,
            "rel_path": str(f.relative_to(ROOT_DIR)).replace("\\", "/"),
            "speaker_id": spk,
            "gender": gender,
            "language": "German",
            "emotion": emo,
            "duration_s": dur
        })
    df = pd.DataFrame(rows)
    emotions = sorted(df["emotion"].unique())
    emo_to_label = {e: i for i, e in enumerate(emotions)}
    df["label"] = df["emotion"].map(emo_to_label)
    return df


def build_cafe_metadata():
    cafe_dir = RAW_DIR / "cafe"
    files = sorted(cafe_dir.glob("*.wav"))
    rows = []
    for f in files:
        # e.g. cafe_01_anger_e9d4b7b83bd1.wav
        parts = f.stem.split("_")
        spk = f"{parts[0]}_{parts[1]}"
        emo = parts[2]
        # Speakers 01, 03, 05, 07, 09, 11 are male; 02, 04, 06, 08, 10, 12 are female
        spk_num = int(parts[1])
        gender = "m" if spk_num % 2 == 1 else "f"
        dur = round(len(load_audio_clip(f)) / SR, 3)
        rows.append({
            "dataset": "cafe",
            "clip_id": f.stem,
            "filename": f.name,
            "rel_path": str(f.relative_to(ROOT_DIR)).replace("\\", "/"),
            "speaker_id": spk,
            "gender": gender,
            "language": "French",
            "emotion": emo,
            "duration_s": dur
        })
    df = pd.DataFrame(rows)
    emotions = sorted(df["emotion"].unique())
    emo_to_label = {e: i for i, e in enumerate(emotions)}
    df["label"] = df["emotion"].map(emo_to_label)
    return df


def build_aesdd_metadata(subset_500=True):
    """
    Build AESDD metadata table strictly aligned with the paper's specification:
    (5 actors x 5 emotions x 20 utterances) = 500 clips (100 per emotion).

    Canonical Structure:
    - 5 emotions: anger, disgust, fear, happiness, sadness.
    - 20 sentences: 01 to 20.
    - Core actors: 1, 2, 3, 4, 5.
    - Official release omissions / corruption:
      * disgust sentence 18 (actor 5 unrecorded) -> substitute take d18 (6)
      * fear sentence 02 (actor 4 unrecorded) -> substitute take f02 (6)
      * happiness sentence 11 (actor 3 unrecorded) -> substitute take h11 (6)
      * sadness sentence 05 (actor 3 corrupted 36-byte file) -> substitute take s05 (6)
    """
    aesdd_dir = RAW_DIR / "aesdd"
    files = sorted(aesdd_dir.rglob("*.wav"))
    
    records_by_emo = {emo: {} for emo in ["anger", "disgust", "fear", "happiness", "sadness"]}
    
    for f in files:
        if f.stat().st_size < 1000:
            # Skip corrupted 36-byte s05 (3).wav
            continue
        emo = f.parent.name.lower()
        if emo not in records_by_emo:
            continue
            
        m = re.match(r'^([a-z])(\d{2})\s*\((0?1|[2-6])\)(.*)$', f.stem)
        if not m:
            continue
        code, sent_str, actor_str, suffix = m.groups()
        sent = int(sent_str)
        actor = int(actor_str)
        if not (1 <= sent <= 20):
            continue
            
        dur = round(len(load_audio_clip(f)) / SR, 3)
        row = {
            "dataset": "aesdd",
            "clip_id": f.stem,
            "filename": f.name,
            "rel_path": str(f.relative_to(ROOT_DIR)).replace("\\", "/"),
            "speaker_id": str(actor),
            "sentence": sent,
            "gender": "unknown",
            "language": "Greek",
            "emotion": emo,
            "duration_s": dur
        }
        
        key = (sent, actor)
        if actor in [1, 2, 3, 4, 5]:
            if key not in records_by_emo[emo] or suffix.strip() == 'a':
                records_by_emo[emo][key] = row
        else:
            # Actor 6 recorded for fallback
            records_by_emo[emo][key] = row

    selected_rows = []
    for emo in ["anger", "disgust", "fear", "happiness", "sadness"]:
        emo_dict = records_by_emo[emo]
        emo_selected = []
        for s in range(1, 21):
            for spk in range(1, 6):
                if (s, spk) in emo_dict:
                    emo_selected.append(emo_dict[(s, spk)])
                elif (s, 6) in emo_dict:
                    emo_selected.append(emo_dict[(s, 6)])
        if subset_500:
            assert len(emo_selected) == 100, f"AESDD {emo} has {len(emo_selected)} clips instead of 100!"
        selected_rows.extend(emo_selected)

    df = pd.DataFrame(selected_rows)
    if subset_500:
        assert len(df) == 500, f"AESDD total clips is {len(df)} instead of 500!"
        for emo, count in df["emotion"].value_counts().items():
            assert count == 100, f"AESDD {emo} has {count} clips instead of 100!"

    emotions = sorted(df["emotion"].unique())
    emo_to_label = {e: i for i, e in enumerate(emotions)}
    df["label"] = df["emotion"].map(emo_to_label)
    return df


def build_mesd_metadata():
    mesd_dir = RAW_DIR / "mesd"
    files = sorted(mesd_dir.glob("*.wav"))
    rows = []
    for f in files:
        # e.g. child_anger_danino_1be6baae.wav
        parts = f.stem.split("_")
        voice_type = parts[0]
        emo = parts[1]
        word = parts[2]
        dur = round(len(load_audio_clip(f)) / SR, 3)
        rows.append({
            "dataset": "mesd",
            "clip_id": f.stem,
            "filename": f.name,
            "rel_path": str(f.relative_to(ROOT_DIR)).replace("\\", "/"),
            "speaker_id": voice_type,  # Voice group: child, female, male
            "gender": voice_type,
            "language": "Spanish",
            "emotion": emo,
            "duration_s": dur
        })
    df = pd.DataFrame(rows)
    emotions = sorted(df["emotion"].unique())
    emo_to_label = {e: i for i, e in enumerate(emotions)}
    df["label"] = df["emotion"].map(emo_to_label)
    return df


def get_metadata(dataset_name):
    """Retrieve or build metadata DataFrame for dataset_name."""
    name = dataset_name.lower().strip()
    proc_csv = PROC_DIR / f"{name}_metadata.csv"
    if proc_csv.is_file():
        return pd.read_csv(proc_csv)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    if name == "emodb":
        df = build_emodb_metadata()
    elif name == "cafe":
        df = build_cafe_metadata()
    elif name == "aesdd":
        df = build_aesdd_metadata(subset_500=True)
    elif name == "mesd":
        df = build_mesd_metadata()
    else:
        raise ValueError(f"Unknown dataset: {name}")

    df.to_csv(proc_csv, index=False)
    return df


if __name__ == "__main__":
    for d in ["emodb", "cafe", "aesdd", "mesd"]:
        df = get_metadata(d)
        print(f"=== {d.upper()} ===")
        print(f"Total rows: {len(df)}")
        print(f"Emotions ({df['emotion'].nunique()}): {dict(df['emotion'].value_counts())}")
        print(f"Speakers ({df['speaker_id'].nunique()}): {sorted(df['speaker_id'].unique().tolist())}")
        print(f"Duration mean: {df['duration_s'].mean():.2f}s (min: {df['duration_s'].min():.2f}s, max: {df['duration_s'].max():.2f}s)")
        print()
