"""
C4: Web-Condition Microphone Simulation and Acoustic Perturbation Engine.
Simulates realistic browser recording conditions:
1. Random microphone gain (-6 dB to +3 dB)
2. Background ambient noise (10 to 30 dB SNR)
3. Light room reverberation (early acoustic reflection taps)
4. Microphone frequency response EQ (low-cut < 80 Hz, high-shelf roll-off > 7000 Hz)
5. Non-linear soft-clipping / pre-amp saturation (tanh)

Deployability assurance: Used to generate mic-simulated test audio for honesty checks,
and optional training-time robustness augmentation (train audio only).
"""

import numpy as np
from scipy import signal

SR = 16000


def simulate_web_microphone(y, sr=SR, rng=None, snr_db=None):
    """
    Apply realistic browser microphone acoustic perturbations to a 1D audio waveform.
    
    Parameters
    ----------
    y : np.ndarray
        1D float32 audio signal.
    sr : int
        Sample rate (default 16000).
    rng : np.random.Generator, optional
        Numpy random generator for reproducibility.
    snr_db : float, optional
        Fixed SNR in dB; if None, chosen uniformly in [12.0, 28.0].
        
    Returns
    -------
    np.ndarray (1D float32)
        Perturbed audio waveform.
    """
    if len(y) == 0:
        return y
    if rng is None:
        rng = np.random.default_rng(42)
        
    y_out = y.copy().astype(np.float32)
    
    # 1. Random gain variation (-6 dB to +3 dB)
    gain_db = rng.uniform(-6.0, 3.0)
    gain_linear = 10.0 ** (gain_db / 20.0)
    y_out = y_out * gain_linear
    
    # 2. Light Room Reverberation (early reflections: 3 taps at ~15ms, 30ms, 45ms)
    taps = [
        (int(0.015 * sr), rng.uniform(0.15, 0.25)),
        (int(0.030 * sr), rng.uniform(0.08, 0.15)),
        (int(0.045 * sr), rng.uniform(0.03, 0.08))
    ]
    reverb = y_out.copy()
    for delay_samples, tap_gain in taps:
        if delay_samples < len(y_out):
            reverb[delay_samples:] += y_out[:-delay_samples] * tap_gain
    y_out = reverb
    
    # 3. Additive Background Noise (10 to 30 dB SNR)
    if snr_db is None:
        target_snr = rng.uniform(12.0, 28.0)
    else:
        target_snr = snr_db
        
    sig_power = np.mean(y_out ** 2)
    if sig_power > 1e-9:
        noise_power = sig_power / (10.0 ** (target_snr / 10.0))
        # Pinkish / colored noise via filtered white noise
        raw_noise = rng.normal(0, np.sqrt(noise_power), size=len(y_out)).astype(np.float32)
        # Low-pass filter noise slightly to simulate room ventilation / computer fan
        b_noise, a_noise = signal.butter(2, 2500 / (sr / 2), btype="low")
        colored_noise = signal.lfilter(b_noise, a_noise, raw_noise).astype(np.float32)
        y_out += colored_noise
        
    # 4. Microphone Frequency Response EQ (Bandpass: cut < 80 Hz, roll-off > 7000 Hz)
    try:
        b_eq, a_eq = signal.butter(2, [80 / (sr / 2), 7000 / (sr / 2)], btype="bandpass")
        y_out = signal.lfilter(b_eq, a_eq, y_out).astype(np.float32)
    except Exception:
        pass
        
    # 5. Non-linear soft-clipping saturation
    peak = np.max(np.abs(y_out)) if len(y_out) > 0 else 0.0
    if peak > 1e-6:
        # Normalize and apply subtle tanh saturation
        y_norm = y_out / peak
        y_out = np.tanh(1.2 * y_norm) / np.tanh(1.2)
        
    return y_out.astype(np.float32)


if __name__ == "__main__":
    y_synth = np.sin(2 * np.pi * 440 * np.linspace(0, 2, 32000)).astype(np.float32)
    y_mic = simulate_web_microphone(y_synth, sr=16000)
    print("Mic simulation test successful!")
    print(f"Orig peak: {np.max(np.abs(y_synth)):.3f}, Mic peak: {np.max(np.abs(y_mic)):.3f}")
