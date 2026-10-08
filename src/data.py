"""Data loading and preprocessing utilities for EmoDB.

Provides:
- load_audio(path): imports raw audio at 16 kHz mono without padding/trimming.
- build_metadata(): extracts metadata (speaker, gender, emotion, label, duration)
  for all EmoDB .wav files and writes data/processed/metadata.csv.
"""

from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from src.config import (
    SAMPLE_RATE,
    DATA_RAW_EMODB_DIR,
    DATA_PROCESSED_DIR,
    EMOTION_CODE_MAP,
    EMOTION_LABEL_MAP,
    SPEAKER_GENDER_MAP,
)

# Expected emotion distributions for EmoDB validation
EXPECTED_EMOTION_COUNTS = {
    "anger": 127,
    "boredom": 81,
    "neutral": 79,
    "happiness": 71,
    "fear": 69,
    "sadness": 62,
    "disgust": 46,
}
EXPECTED_TOTAL_FILES = 535


def load_audio(path: Union[str, Path], sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file as a mono waveform at the given sample rate.

    Does NOT apply any artificial padding or trimming to the waveform.

    Parameters
    ----------
    path : str or Path
        Path to the audio file (.wav).
    sr : int, optional
        Target sample rate in Hz (default: config.SAMPLE_RATE = 16000).

    Returns
    -------
    np.ndarray
        1D float32 numpy array representing the raw audio waveform.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # librosa.load converts multi-channel to mono (mono=True) and resamples to sr
    audio, _ = librosa.load(str(path), sr=sr, mono=True)
    return audio


def build_metadata(
    raw_dir: Path = DATA_RAW_EMODB_DIR,
    output_csv: Path = DATA_PROCESSED_DIR / "metadata.csv",
) -> pd.DataFrame:
    """Parse EmoDB audio filenames, compute durations, and save metadata.csv.

    Columns produced:
    - filename: Base name of the file (e.g. '03a01Fa.wav')
    - speaker: Two-digit speaker identifier (e.g. '03')
    - gender: Speaker gender ('m' or 'f')
    - emotion: Emotion name in lower case ('happiness', 'anger', etc.)
    - label: Numeric label from 0 to 6 (assigned alphabetically)
    - duration_s: Total audio duration in seconds rounded to 4 decimals

    Parameters
    ----------
    raw_dir : Path
        Directory containing the raw EmoDB .wav files.
    output_csv : Path
        Output destination for the metadata CSV.

    Returns
    -------
    pd.DataFrame
        The generated metadata DataFrame.
    """
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Raw EmoDB directory does not exist: {raw_dir}"
        )

    # 1. Collect and alphabetically sort all .wav files by filename
    wav_files = sorted(
        [p for p in raw_dir.iterdir() if p.suffix.lower() == ".wav"],
        key=lambda p: p.name,
    )

    if not wav_files:
        raise ValueError(f"No .wav files found in {raw_dir}")

    records = []
    print(f"Reading {len(wav_files)} audio files from {raw_dir}...")

    # 2. Iterate through each audio file with a progress bar
    for file_path in tqdm(wav_files, desc="Processing audio metadata"):
        fname = file_path.name

        # Characters 1-2: speaker ID (index 0 and 1)
        speaker = fname[0:2]
        if speaker not in SPEAKER_GENDER_MAP:
            raise ValueError(f"Unknown speaker '{speaker}' in file: {fname}")
        gender = SPEAKER_GENDER_MAP[speaker]

        # Character 6: emotion letter (index 5)
        emotion_char = fname[5]
        if emotion_char not in EMOTION_CODE_MAP:
            raise ValueError(
                f"Unknown emotion character '{emotion_char}' at index 5 in file: {fname}"
            )
        emotion = EMOTION_CODE_MAP[emotion_char]
        label = EMOTION_LABEL_MAP[emotion]

        # Load audio using the project's load_audio function and measure duration
        audio = load_audio(file_path, sr=SAMPLE_RATE)
        duration_s = round(len(audio) / SAMPLE_RATE, 4)

        records.append({
            "filename": fname,
            "speaker": speaker,
            "gender": gender,
            "emotion": emotion,
            "label": label,
            "duration_s": duration_s,
        })

    df = pd.DataFrame(records)

    # 3. Validation Checks
    total_rows = len(df)
    print(f"\nTotal rows extracted: {total_rows}")

    if total_rows != EXPECTED_TOTAL_FILES:
        raise AssertionError(
            f"Row count mismatch! Expected {EXPECTED_TOTAL_FILES}, but got {total_rows}."
        )

    # Count samples per emotion
    actual_counts = df["emotion"].value_counts().to_dict()

    print("\n--- Emotion Distribution ---")
    for emotion, expected_count in EXPECTED_EMOTION_COUNTS.items():
        actual_count = actual_counts.get(emotion, 0)
        status = "OK" if actual_count == expected_count else "MISMATCH"
        print(f"  {emotion:<10}: actual={actual_count:>3}, expected={expected_count:>3} [{status}]")

    # Verify that every emotion matches exactly
    for emotion, expected_count in EXPECTED_EMOTION_COUNTS.items():
        actual_count = actual_counts.get(emotion, 0)
        if actual_count != expected_count:
            raise AssertionError(
                f"Emotion count mismatch for '{emotion}'! Expected {expected_count}, got {actual_count}."
            )

    # 4. Save to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nMetadata successfully saved to: {output_csv}")

    return df


if __name__ == "__main__":
    build_metadata()
