import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa
from config import *
from data import load_audio

RESULTS_DIR.mkdir(exist_ok=True)
df = pd.read_csv(PROC_DIR / "metadata.csv")

# 1) class distribution
counts = df["emotion"].value_counts()
plt.figure(figsize=(7, 4))
plt.bar(counts.index, counts.values)
plt.title("EmoDB: clips per emotion")
plt.ylabel("number of clips")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "class_distribution.png", dpi=150)
plt.close()

# 2) MFCC vs MFMC examples for 3 emotions (30 coefficients)
emos = ["anger", "sadness", "happiness"]
fig, axes = plt.subplots(2, 3, figsize=(13, 6))
for j, emo in enumerate(emos):
    fn = df[df.emotion == emo].iloc[0]["filename"]
    y = load_audio(RAW_DIR / fn)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=30, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    mfmc = np.log10(librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=30) @ S + 1e-10)
    axes[0, j].imshow(mfcc, aspect="auto", origin="lower")
    axes[0, j].set_title(f"MFCC - {emo} ({fn})", fontsize=9)
    axes[1, j].imshow(mfmc, aspect="auto", origin="lower")
    axes[1, j].set_title(f"MFMC - {emo}", fontsize=9)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "mfcc_vs_mfmc_examples.png", dpi=150)
plt.close()
print("plots saved")
