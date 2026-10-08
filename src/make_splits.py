import pandas as pd
from sklearn.model_selection import train_test_split
from config import *

df = pd.read_csv(PROC_DIR / "metadata.csv")
idx = df.index.values
tr, tmp = train_test_split(idx, test_size=0.4, stratify=df["label"], random_state=SEED)
va, te = train_test_split(tmp, test_size=0.5, stratify=df.loc[tmp, "label"], random_state=SEED)
df["split"] = "train"
df.loc[va, "split"] = "val"
df.loc[te, "split"] = "test"
df.to_csv(PROC_DIR / "splits.csv", index=False)
print(df["split"].value_counts())
