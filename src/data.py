import sys
import librosa
import pandas as pd
from pathlib import Path
from config import *


def parse_filename(fname):
    stem = Path(fname).stem          # e.g. 03a01Fa
    speaker = stem[0:2]
    code = stem[5]                   # 6th character = emotion letter
    return {"filename": Path(fname).name, "speaker": speaker,
            "gender": SPEAKER_GENDER[speaker], "emotion": EMOTION_MAP[code]}


def load_audio(path):
    """Load a clip at 16 kHz mono. No padding or trimming."""
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y


def build_metadata():
    files = sorted(RAW_DIR.glob("*.wav"))   # sorted = fixed row order
    if len(files) != N_ROWS:
        print("ERROR: expected", N_ROWS, "wav files, found", len(files))
        sys.exit(1)
    rows = []
    for f in files:
        row = parse_filename(f.name)
        row["duration_s"] = round(len(load_audio(f)) / SR, 3)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["label"] = df["emotion"].astype("category").cat.codes
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROC_DIR / "metadata.csv", index=False)
    return df


if __name__ == "__main__":
    df = build_metadata()
    print(df.shape)
    print(df["emotion"].value_counts())
