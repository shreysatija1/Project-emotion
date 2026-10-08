import numpy as np
import pandas as pd
from config import *


def save_features(kind, n, X):
    assert X.shape[0] == N_ROWS, f"expected {N_ROWS} rows, got {X.shape[0]}"
    FEAT_DIR.mkdir(exist_ok=True)
    np.save(FEAT_DIR / f"emodb_{kind}_{n}.npy", X)


def load_features(kind, n):
    return np.load(FEAT_DIR / f"emodb_{kind}_{n}.npy")


def load_split():
    df = pd.read_csv(PROC_DIR / "splits.csv")
    return (df.index[df.split == "train"].values,
            df.index[df.split == "val"].values,
            df.index[df.split == "test"].values,
            df["label"].values)
