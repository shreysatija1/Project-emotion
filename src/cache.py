import numpy as np
import pandas as pd
from config import *


def save_features(kind, n, X):
    assert X.shape[0] == N_ROWS, f"expected {N_ROWS} rows, got {X.shape[0]}"
    FEAT_DIR.mkdir(exist_ok=True)
    np.save(FEAT_DIR / f"emodb_{kind}_{n}.npy", X)


def load_features(kind, n):
    return np.load(FEAT_DIR / f"emodb_{kind}_{n}.npy")


def load_split(version=None):
    if version is None:
        df = pd.read_csv(PROC_DIR / "splits.csv")
    else:
        v = str(version).lower().strip()
        if not v.startswith("version_"):
            v = f"version_{v}"
        p = PROC_DIR / f"{v}_metadata.csv"
        if not p.is_file():
            p = FEAT_DIR / f"{v}_metadata.csv"
        df = pd.read_csv(p)
    return (df.index[df.split == "train"].values,
            df.index[df.split == "val"].values,
            df.index[df.split == "test"].values,
            df["label"].values)
