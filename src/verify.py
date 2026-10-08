import random
import numpy as np
import pandas as pd
from config import *
from data import load_audio
from cache import load_features, load_split
from features_mfcc import extract_mfcc
from features_mfmc import extract_mfmc

tr, va, te, y = load_split()
assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te)), "split overlap"
assert len(tr) + len(va) + len(te) == N_ROWS
print("splits OK:", len(tr), len(va), len(te))

for n in COEFF_LIST:
    a, b = load_features("mfcc", n), load_features("mfmc", n)
    assert a.shape == b.shape == (N_ROWS, 2 * n), (n, a.shape, b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all(), f"NaN/inf at {n}"
    print("shapes OK for", n, a.shape)

# spot-check: recompute 3 random rows and compare with the saved rows
meta = pd.read_csv(PROC_DIR / "metadata.csv")
random.seed(SEED)
for i in random.sample(range(N_ROWS), 3):
    audio = load_audio(RAW_DIR / meta.loc[i, "filename"])
    assert np.allclose(extract_mfcc(audio, SR, 30), load_features("mfcc", 30)[i], atol=1e-5)
    assert np.allclose(extract_mfmc(audio, SR, 30), load_features("mfmc", 30)[i], atol=1e-5)
    print("row", i, meta.loc[i, "filename"], "matches saved features")
print("ALL CHECKS PASSED")
