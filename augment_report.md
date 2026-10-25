# Augmentation and Dataset Verification Report

## 1. Augmentation Protocol Specification

We implemented and verified two experimental augmentation protocols:

1. **Version A (After-Split / Split-then-Augment)**:
   - Takes the 60/20/20 stratified split established in Week 2 (`splits.csv`):
     - Train: 321 clips
     - Val: 107 clips
     - Test: 107 clips
   - Augments **only** the 321 training clips into 5 sets ($5 \times 321 = 1,605$ rows).
   - Keeps validation and test sets strictly unaugmented ($1 \times 107 = 107$ rows each, `set1` clean audio only).
   - Total rows: $1,605 + 107 + 107 = \mathbf{1,819}$ rows.
   - **Leakage**: Exactly **0%** overlap between training source clips and validation/test source clips.

2. **Version B (Before-Split / Augment-then-Split)**:
   - Augments all 535 clips in EmoDB into 5 sets ($5 \times 535 = \mathbf{2,675}$ rows).
   - Performs a new stratified 60/20/20 split across the resulting 2,675 rows:
     - Train: 1,605 rows (60%)
     - Val: 535 rows (20%)
     - Test: 535 rows (20%)
   - **Leakage**: **96.64%** of validation rows and **95.89%** of test rows have their source recording present in the training split.

---

## 2. Noise Generation & Set Structure

For each source recording waveform $x[n]$, five representations are generated:
- **Set 1**: Original audio waveform ($x[n]$), noise level $\sigma = 0.0$.
- **Set 2**: Zero-noise copy ($x[n]$), noise level $\sigma = 0.0$.
- **Set 3**: $x[n] + 0.01 \cdot \mathcal{N}(0, 1)$, noise level $\sigma = 0.01$.
- **Set 4**: $x[n] + 0.025 \cdot \mathcal{N}(0, 1)$, noise level $\sigma = 0.025$.
- **Set 5**: $x[n] + 0.05 \cdot \mathcal{N}(0, 1)$, noise level $\sigma = 0.05$.

RNG initialized with seed `42` (`np.random.default_rng(42)`).

---

## 3. Automated Verification Checklist

All automated tests executed on the saved artifacts passed:

| Test Name | Description | Result |
| :--- | :--- | :---: |
| `test_version_a_no_source_clip_crosses_splits` | Verifies `train_sources & val_sources == 0` and `train_sources & test_sources == 0` | **PASS** |
| `test_version_a_val_test_set1_only` | Verifies all val and test rows in Version A are strictly `set1` (clean audio) | **PASS** |
| `test_version_a_class_counts_5x` | Verifies Version A train emotion counts equal exactly $5 \times$ the Week 2 train counts | **PASS** |
| `test_version_a_val_test_counts_match_week2` | Verifies Version A val and test emotion counts match Week 2 counts exactly | **PASS** |
| `test_no_nans_in_saved_artifacts` | Verifies no NaN or Inf values in any feature matrices, labels, or metadata | **PASS** |
| `test_correct_shapes` | Verifies dimensions: Version A is $(1819, 2n)$, Version B is $(2675, 2n)$ for $n \in \{12, 24, 30\}$ | **PASS** |
| `test_set2_features_equal_set1_features_version_a` | Verifies zero-noise copy features match original features bitwise in Version A | **PASS** |
| `test_set2_features_equal_set1_features_version_b` | Verifies zero-noise copy features match original features bitwise in Version B | **PASS** |
| `test_version_b_stratified_split_ratios` | Verifies 60/20/20 split proportions and representation across all 7 emotions | **PASS** |
