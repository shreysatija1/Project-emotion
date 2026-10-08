import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from config import *
from data import load_audio
from cache import save_features


def extract_mfcc(y, sr, n_coeff):
    """MFCC pooled over time: mean and std of each coefficient -> 2*n_coeff values."""
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_coeff, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return np.concatenate([m.mean(axis=1), m.std(axis=1)])


if __name__ == "__main__":
    df = pd.read_csv(PROC_DIR / "metadata.csv")
    feats = {n: [] for n in COEFF_LIST}
    for fn in tqdm(df["filename"]):
        y = load_audio(RAW_DIR / fn)
        for n in COEFF_LIST:
            feats[n].append(extract_mfcc(y, SR, n))
    for n in COEFF_LIST:
        X = np.vstack(feats[n])
        save_features("mfcc", n, X)
        print("saved mfcc", n, X.shape)
