"""
Acoustic Augmentation Module for Week 5.

Implements:
1. Pitch Shifting:
   - Shifts waveform by ±2 semitones: [0, -2, -1, +1, +2] (5 copies).
   - Fast, high-fidelity resampling using librosa soxr_mq.
2. Time Stretching:
   - Stretches waveform by rates 0.9x to 1.1x: [1.0, 0.90, 0.95, 1.05, 1.10] (5 copies).
3. SpecAugment (Time & Frequency Masking):
   - Applied to the 2D frame-level spectrogram (n_coeff, T) BEFORE mean+std pooling.
   - Frequency masking: zeroes consecutive frequency bins [f0, f0+f].
   - Time masking: zeroes consecutive time frames [t0, t0+t].
   - 5 copies per clip: 1 unmasked + 4 distinct masked versions.
"""

import numpy as np
import librosa

from features_144 import extract_raw_mfcc, extract_raw_mfmc


def augment_pitch_shift(y, sr=16000, steps=(0, -2, -1, 1, 2)):
    """
    Generate 5 pitch-shifted copies of waveform y.
    Returns list of 5 waveforms (float32).
    """
    copies = []
    for step in steps:
        if step == 0:
            copies.append(y.astype(np.float32).copy())
        else:
            y_shift = librosa.effects.pitch_shift(
                y.astype(np.float32), sr=sr, n_steps=step, res_type="soxr_mq"
            )
            copies.append(y_shift.astype(np.float32))
    return copies


def augment_time_stretch(y, rates=(1.0, 0.90, 0.95, 1.05, 1.10)):
    """
    Generate 5 time-stretched copies of waveform y.
    Returns list of 5 waveforms (float32).
    """
    copies = []
    for rate in rates:
        if rate == 1.0:
            copies.append(y.astype(np.float32).copy())
        else:
            y_stretch = librosa.effects.time_stretch(y.astype(np.float32), rate=rate)
            copies.append(y_stretch.astype(np.float32))
    return copies


def apply_spec_augment(spec, rng, max_mask_pct_freq=0.20, max_mask_pct_time=0.15):
    """
    Apply one frequency mask and one time mask to 2D matrix spec (n_coeff, T).
    Mask values replaced with channel mean.
    """
    masked = spec.copy()
    n_coeff, T = masked.shape

    # Frequency masking
    max_f = max(1, int(n_coeff * max_mask_pct_freq))
    f = rng.integers(1, max_f + 1)
    f0 = rng.integers(0, n_coeff - f + 1)
    # Mask with channel mean
    masked[f0:f0 + f, :] = 0.0

    # Time masking
    if T > 5:
        max_t = max(1, int(T * max_mask_pct_time))
        t = rng.integers(1, max_t + 1)
        t0 = rng.integers(0, T - t + 1)
        masked[:, t0:t0 + t] = 0.0

    return masked


def extract_spec_augment_features(y, sr, feat_kind, n_coeff, rng, n_copies=5):
    """
    Extract 5 SpecAugment feature vectors (2*n_coeff,) by masking BEFORE mean+std pooling.
    Copy 0: unmasked baseline.
    Copies 1..4: random SpecAugment masks.
    """
    if feat_kind == "mfcc":
        raw = extract_raw_mfcc(y, sr, n_coeff)
    elif feat_kind == "mfmc":
        raw = extract_raw_mfmc(y, sr, n_coeff)
    else:
        raise ValueError(f"Unknown feature kind: {feat_kind}")

    copies = []
    # Copy 0: unmasked
    feat0 = np.concatenate([raw.mean(axis=1), raw.std(axis=1, ddof=1) if raw.shape[1] > 1 else np.zeros(n_coeff)]).astype(np.float32)
    copies.append(feat0)

    # Copies 1..4
    for _ in range(n_copies - 1):
        masked_raw = apply_spec_augment(raw, rng)
        std_val = masked_raw.std(axis=1, ddof=1) if masked_raw.shape[1] > 1 else np.zeros(n_coeff)
        feat_m = np.concatenate([masked_raw.mean(axis=1), std_val]).astype(np.float32)
        copies.append(feat_m)

    return copies
