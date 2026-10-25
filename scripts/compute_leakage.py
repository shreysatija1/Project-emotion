import pandas as pd

meta_b = pd.read_csv("data/processed/version_b_metadata.csv")
tr_sources = set(meta_b[meta_b["split"] == "train"]["source_clip_id"])
val_df = meta_b[meta_b["split"] == "val"]
test_df = meta_b[meta_b["split"] == "test"]

val_leaked_rows = val_df["source_clip_id"].isin(tr_sources)
test_leaked_rows = test_df["source_clip_id"].isin(tr_sources)

n_val = len(val_df)
n_test = len(test_df)
val_leak_count = int(val_leaked_rows.sum())
test_leak_count = int(test_leaked_rows.sum())

val_leak_pct = 100.0 * val_leak_count / n_val
test_leak_pct = 100.0 * test_leak_count / n_test

val_sources = set(val_df["source_clip_id"])
test_sources = set(test_df["source_clip_id"])

val_leaked_sources = val_sources & tr_sources
test_leaked_sources = test_sources & tr_sources

print(f"Total Version B rows: {len(meta_b)}")
print(f"Train rows: {len(meta_b[meta_b['split'] == 'train'])}")
print(f"Unique source clips in train: {len(tr_sources)} / 535 ({100.0 * len(tr_sources) / 535:.2f}%)")
print(f"Val rows total: {n_val}")
print(f"Val rows with source_clip_id in train: {val_leak_count} / {n_val} ({val_leak_pct:.2f}%)")
print(f"Unique val sources in train: {len(val_leaked_sources)} / {len(val_sources)} ({100.0 * len(val_leaked_sources) / len(val_sources):.2f}%)")
print(f"Test rows total: {n_test}")
print(f"Test rows with source_clip_id in train: {test_leak_count} / {n_test} ({test_leak_pct:.2f}%)")
print(f"Unique test sources in train: {len(test_leaked_sources)} / {len(test_sources)} ({100.0 * len(test_leaked_sources) / len(test_sources):.2f}%)")
