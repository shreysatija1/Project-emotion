"""
Feature extraction modules for 144-input representations.

Investigating the paper discrepancy:
The paper ("Data augmentation using a 1D-CNN model with MFCC/MFMC features for speech emotion recognition",
Automatika 2024) states:
"One hundred forty-four feature vectors as input data for our CNN's first layer."

We implement the two primary plausible interpretations:
1. 'frames144' (Temporal Sequence):
   Extracts MFCC/MFMC with n_coeff (12, 24, 30) across time frames.
   The time dimension is fixed to exactly 144 frames (zero-padding if shorter, center-cropping if longer).
   Input shape to 1D-CNN: (144, n_coeff).
   Convolution filters (kernel 15) slide temporally across the 144 time frames.
   4 MaxPool1D(2) layers reduce 144 -> 72 -> 36 -> 18 -> 9 frames cleanly, yielding 9*64 = 576 units into Dense(120).

2. 'stats144' (Extended Statistical Aggregation):
   Extracts MFCC/MFMC with static, delta (velocity), and delta-delta (acceleration) trajectories across time.
   Computes summary statistics across time to form a 144-dimensional vector:
   - For 24 coeffs: 24 static + 24 delta + 24 delta2 = 72 trajectories.
     Mean and Std per trajectory = 72 * 2 = 144 features.
   - For 12 coeffs: 12 static + 12 delta + 12 delta2 = 36 trajectories.
     Mean, Std, Min, Max per trajectory = 36 * 4 = 144 features.
   - For 30 coeffs: 30 static + 30 delta + 30 delta2 = 90 trajectories.
     Mean and Std per trajectory = 180 features, truncated to 144 features.
   Input shape to 1D-CNN: (144, 1).
"""

import numpy as np
import librosa
from pathlib import Path

N_FFT = 512
HOP_LENGTH = 256
TARGET_FRAMES = 144


def extract_raw_mfcc(y, sr, n_coeff):
    """Raw MFCC spectrogram of shape (n_coeff, T)."""
    return librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_coeff, n_fft=N_FFT, hop_length=HOP_LENGTH
    )


def extract_raw_mfmc(y, sr, n_coeff):
    """Raw MFMC spectrogram of shape (n_coeff, T)."""
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    mel = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=n_coeff)
    return np.log10(mel @ S + 1e-10)


def fix_frames(spec, target_frames=TARGET_FRAMES):
    """
    Adjust spectrogram along time axis to exactly target_frames.
    spec shape: (n_coeff, T)
    Returns: (target_frames, n_coeff)
    """
    n_coeff, T = spec.shape
    if T == target_frames:
        out = spec
    elif T < target_frames:
        pad_width = target_frames - T
        out = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    else:
        # Center crop
        start = (T - target_frames) // 2
        out = spec[:, start:start + target_frames]
    
    # Transpose so time is the first dimension: (target_frames, n_coeff)
    return out.T.astype(np.float32)


def extract_mfcc_frames144(y, sr, n_coeff):
    """Extract MFCC as temporal frame sequence of shape (144, n_coeff)."""
    raw = extract_raw_mfcc(y, sr, n_coeff)
    return fix_frames(raw, TARGET_FRAMES)


def extract_mfmc_frames144(y, sr, n_coeff):
    """Extract MFMC as temporal frame sequence of shape (144, n_coeff)."""
    raw = extract_raw_mfmc(y, sr, n_coeff)
    return fix_frames(raw, TARGET_FRAMES)


def extract_mfcc_stats144(y, sr, n_coeff):
    """
    Extract MFCC with static, delta, and delta-delta aggregated to exactly 144 scalar features.
    Returns: vector of shape (144,)
    """
    raw = extract_raw_mfcc(y, sr, n_coeff)
    d1 = librosa.feature.delta(raw, order=1)
    d2 = librosa.feature.delta(raw, order=2)
    
    if n_coeff == 24:
        # 24 * 3 = 72 trajectories. Mean and Std = 144 features.
        feats = np.concatenate([
            raw.mean(axis=1), raw.std(axis=1),
            d1.mean(axis=1), d1.std(axis=1),
            d2.mean(axis=1), d2.std(axis=1)
        ])
    elif n_coeff == 12:
        # 12 * 3 = 36 trajectories. Mean, Std, Min, Max = 144 features.
        feats = np.concatenate([
            raw.mean(axis=1), raw.std(axis=1), raw.min(axis=1), raw.max(axis=1),
            d1.mean(axis=1), d1.std(axis=1), d1.min(axis=1), d1.max(axis=1),
            d2.mean(axis=1), d2.std(axis=1), d2.min(axis=1), d2.max(axis=1)
        ])
    elif n_coeff == 30:
        # 30 * 3 = 90 trajectories. Mean and Std = 180 features -> truncate first 144.
        full_feats = np.concatenate([
            raw.mean(axis=1), raw.std(axis=1),
            d1.mean(axis=1), d1.std(axis=1),
            d2.mean(axis=1), d2.std(axis=1)
        ])
        feats = full_feats[:144]
    else:
        raise ValueError(f"Unsupported n_coeff for stats144: {n_coeff}")
    
    return feats.astype(np.float32)


def extract_mfmc_stats144(y, sr, n_coeff):
    """
    Extract MFMC with static, delta, and delta-delta aggregated to exactly 144 scalar features.
    Returns: vector of shape (144,)
    """
    raw = extract_raw_mfmc(y, sr, n_coeff)
    d1 = librosa.feature.delta(raw, order=1)
    d2 = librosa.feature.delta(raw, order=2)
    
    if n_coeff == 24:
        feats = np.concatenate([
            raw.mean(axis=1), raw.std(axis=1),
            d1.mean(axis=1), d1.std(axis=1),
            d2.mean(axis=1), d2.std(axis=1)
        ])
    elif n_coeff == 12:
        feats = np.concatenate([
            raw.mean(axis=1), raw.std(axis=1), raw.min(axis=1), raw.max(axis=1),
            d1.mean(axis=1), d1.std(axis=1), d1.min(axis=1), d1.max(axis=1),
            d2.mean(axis=1), d2.std(axis=1), d2.min(axis=1), d2.max(axis=1)
        ])
    elif n_coeff == 30:
        full_feats = np.concatenate([
            raw.mean(axis=1), raw.std(axis=1),
            d1.mean(axis=1), d1.std(axis=1),
            d2.mean(axis=1), d2.std(axis=1)
        ])
        feats = full_feats[:144]
    else:
        raise ValueError(f"Unsupported n_coeff for stats144: {n_coeff}")
    
    return feats.astype(np.float32)
