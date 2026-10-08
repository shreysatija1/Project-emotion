import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from config import *
from data import load_audio
from cache import save_features


def extract_mfmc(y, sr, n_coeff):
    """MFMC = log10 of mel-filterbank-weighted MAGNITUDE spectrum (not squared),
    no DCT. Pooled over time: mean and std -> 2*n_coeff values."""
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    mel = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=n_coeff)
    m = np.log10(mel @ S + 1e-10)
    return np.concatenate([m.mean(axis=1), m.std(axis=1)])


if __name__ == "__main__":
    df = pd.read_csv(PROC_DIR / "metadata.csv")
    feats = {n: [] for n in COEFF_LIST}
    for fn in tqdm(df["filename"]):
        y = load_audio(RAW_DIR / fn)
        for n in COEFF_LIST:
            feats[n].append(extract_mfmc(y, SR, n))
    for n in COEFF_LIST:
        X = np.vstack(feats[n])
        save_features("mfmc", n, X)
        print("saved mfmc", n, X.shape)
