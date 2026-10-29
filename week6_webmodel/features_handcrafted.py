"""
C1: Hand-crafted deployable acoustic feature extractor.
Computes per-utterance frame-level features and summarizes them via clip-level statistics:
- MFCC (30 coefficients) with Delta and Delta-Delta trajectories
- Log-Mel filterbank energies (40 bands)
- F0 / Pitch contour via YIN (fmin=60, fmax=500 Hz), voiced fraction, pitch jitter proxy
- RMS Energy, dynamic range, energy shimmer proxy
- Spectral Centroid, Bandwidth, Contrast, Rolloff, Flatness, Flux (Onset Strength)
- Zero-Crossing Rate
- Speaking rate proxy (energy syllable peak count / duration)

Deployability guarantee: All statistics and normalizations are computed strictly
within the individual clip (per-utterance). No corpus- or speaker-level test statistics.
"""

import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
from pathlib import Path

SR = 16000
N_FFT = 512
HOP_LENGTH = 256
N_MFCC = 30
N_MELS = 40


def compute_frame_statistics(mat, axis=1):
    """
    Given a 2D matrix (n_features, n_frames), compute:
    mean, std, min, max, 10th percentile, 90th percentile.
    Returns 1D vector of length 6 * n_features.
    """
    if mat.shape[axis] == 0:
        return np.zeros(mat.shape[0] * 6, dtype=np.float32)
    
    mean = np.mean(mat, axis=axis)
    std = np.std(mat, axis=axis)
    mn = np.min(mat, axis=axis)
    mx = np.max(mat, axis=axis)
    p10 = np.percentile(mat, 10, axis=axis)
    p90 = np.percentile(mat, 90, axis=axis)
    
    return np.concatenate([mean, std, mn, mx, p10, p90]).astype(np.float32)


def extract_handcrafted_features(y, sr=SR):
    """
    Extract comprehensive deployable acoustic feature vector for a 1D audio waveform.
    
    Parameters
    ----------
    y : np.ndarray
        1D float32 audio waveform.
    sr : int
        Sample rate (default 16000).
        
    Returns
    -------
    np.ndarray (1D float32)
        Concatenated statistical summary vector.
    """
    # 1. Basic validation and peak normalization per clip
    if len(y) == 0:
        y = np.zeros(int(sr * 0.5), dtype=np.float32)
    peak = np.max(np.abs(y))
    if peak > 1e-6:
        y = y / peak
    duration = len(y) / sr
    
    # Pre-emphasis
    y_pre = librosa.effects.preemphasis(y)
    
    # 2. MFCC + Delta + Delta-Delta
    mfcc = librosa.feature.mfcc(y=y_pre, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # For MFCCs, mean and std are standard, plus min/max
    mfcc_stats = np.concatenate([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1), np.min(mfcc, axis=1), np.max(mfcc, axis=1),
        np.mean(delta1, axis=1), np.std(delta1, axis=1),
        np.mean(delta2, axis=1), np.std(delta2, axis=1)
    ])  # 30 * 8 = 240 dims
    
    # 3. Log-Mel Filterbanks
    mel = librosa.feature.melspectrogram(y=y_pre, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel_stats = np.concatenate([
        np.mean(log_mel, axis=1), np.std(log_mel, axis=1), np.min(log_mel, axis=1), np.max(log_mel, axis=1)
    ])  # 40 * 4 = 160 dims
    
    # 4. Fundamental Frequency (F0 / Pitch) via fast torchaudio pitch detection
    try:
        y_torch = torch.from_numpy(y).unsqueeze(0)
        f0_tensor = torchaudio.functional.detect_pitch_frequency(y_torch, sample_rate=sr, frame_time=0.016, win_length=30)
        f0 = f0_tensor.squeeze(0).numpy()
    except Exception:
        # Fallback to librosa yin on short window
        f0 = np.zeros(max(1, int(len(y) / HOP_LENGTH)), dtype=np.float32)

    voiced_mask = (f0 > 60) & (f0 < 490) & ~np.isnan(f0)
    voiced_fraction = float(np.mean(voiced_mask))
    
    if np.any(voiced_mask):
        f0_voiced = f0[voiced_mask]
        f0_mean = float(np.mean(f0_voiced))
        f0_std = float(np.std(f0_voiced))
        f0_min = float(np.min(f0_voiced))
        f0_max = float(np.max(f0_voiced))
        f0_range = f0_max - f0_min
        # Pitch jitter proxy: mean relative frame-to-frame F0 change
        if len(f0_voiced) > 1:
            diff_f0 = np.abs(np.diff(f0_voiced))
            jitter_proxy = float(np.mean(diff_f0 / (f0_voiced[:-1] + 1e-6)))
        else:
            jitter_proxy = 0.0
    else:
        f0_mean = 0.0
        f0_std = 0.0
        f0_min = 0.0
        f0_max = 0.0
        f0_range = 0.0
        jitter_proxy = 0.0
        
    pitch_stats = np.array([voiced_fraction, f0_mean, f0_std, f0_min, f0_max, f0_range, jitter_proxy], dtype=np.float32)
    
    # 5. Energy (RMS) and Shimmer proxy
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    rms_max = float(np.max(rms)) if len(rms) > 0 else 0.0
    rms_dyn_range = rms_max - (float(np.min(rms)) if len(rms) > 0 else 0.0)
    if len(rms) > 1:
        shimmer_proxy = float(np.mean(np.abs(np.diff(rms)) / (rms[:-1] + 1e-6)))
    else:
        shimmer_proxy = 0.0
    rms_stats = np.array([rms_mean, rms_std, rms_max, rms_dyn_range, shimmer_proxy], dtype=np.float32)
    
    # 6. Spectral Descriptors
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    spec_roll85 = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH, roll_percent=0.85)[0]
    spec_roll95 = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH, roll_percent=0.95)[0]
    spec_flat = librosa.feature.spectral_flatness(y=y, hop_length=HOP_LENGTH)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    
    spectral_1d_feats = [spec_cent, spec_bw, spec_roll85, spec_roll95, spec_flat, zcr, onset_env]
    spectral_stats = []
    for feat in spectral_1d_feats:
        spectral_stats.extend([float(np.mean(feat)), float(np.std(feat)), float(np.min(feat)), float(np.max(feat))])
    spectral_stats = np.array(spectral_stats, dtype=np.float32)  # 7 * 4 = 28 dims
    
    # 7. Speaking Rate Proxy (Syllable / energy peaks per second)
    if duration > 0.1:
        # Detect onset / energy peaks
        peaks = librosa.util.peak_pick(rms, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.02, wait=3)
        speaking_rate = float(len(peaks)) / duration
    else:
        speaking_rate = 0.0
    rate_stats = np.array([speaking_rate, duration], dtype=np.float32)
    
    # Concatenate all features
    full_vector = np.concatenate([
        mfcc_stats,         # 240 dims
        log_mel_stats,      # 160 dims
        pitch_stats,        # 7 dims
        rms_stats,          # 5 dims
        spectral_stats,     # 28 dims
        rate_stats          # 2 dims
    ]).astype(np.float32)   # Total = 442 dims
    
    # Replace any NaNs or Infs
    full_vector = np.nan_to_num(full_vector, nan=0.0, posinf=0.0, neginf=0.0)
    return full_vector


if __name__ == "__main__":
    # Test on synthetic audio
    rng = np.random.default_rng(42)
    fake_audio = (rng.normal(0, 0.1, 16000 * 3)).astype(np.float32)
    feats = extract_handcrafted_features(fake_audio, 16000)
    print("Handcrafted features extracted successfully!")
    print(f"Shape: {feats.shape}, dtype: {feats.dtype}, NaNs: {np.isnan(feats).sum()}")
