# Verification Report: Integrity, Timestamps, and Model Auditing

This report provides evidence verifying training isolation, timestamps, loader correctness, overfitting observations, and potential anomalies in Week 3 experiments.

---

### 1. Evidence of File Timestamps & Evaluation Sequence

To prove that models were selected and frozen **strictly on validation accuracy before any test set evaluation took place**, the file system timestamps on disk are:

```powershell
Get-Item selected_models.md, results/final_results_table.csv, results/final_results_table.png, results/confusion_matrix_best_mfcc_after_split.png | Select-Object Name, CreationTime, LastWriteTime | Format-List
```

**Recorded Timestamps**:
- `selected_models.md`:
  - **Creation Time**: `23-10-2026 19:15:35`
  - **Last Write Time**: `23-10-2026 19:15:35`
- `final_results_table.csv`:
  - **Creation Time**: `23-10-2026 19:16:44`
  - **Last Write Time**: `23-10-2026 19:16:44`
- `final_results_table.png`:
  - **Creation Time**: `23-10-2026 19:16:45`
  - **Last Write Time**: `23-10-2026 19:16:45`
- `confusion_matrix_best_mfcc_after_split.png`:
  - **Creation Time**: `23-10-2026 19:16:46`
  - **Last Write Time**: `23-10-2026 19:16:46`

**Confirmation**:
`selected_models.md` was written and closed at **19:15:35**, exactly **1 minute and 9 seconds before** `final_results_table.csv` was created (19:16:44). In [`src/evaluate.py`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/src/evaluate.py#L429-L436), `write_and_freeze_selected_models()` is called on line 429 to select checkpoints purely from validation JSON summaries, and `evaluate_test_set()` is executed on line 436 only after `selected_models.md` is committed to disk.

---

### 2. Exact Lines in `src/train.py` for Scaler Fitting and Data Loading

#### (a) Scaler Fitting (Training Data Only)
- **File**: [`src/train.py`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/src/train.py#L164-L168)
- **Lines 164–168**:
  ```python
  # 2. Standardize with Train Statistics ONLY
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_val_scaled = scaler.transform(X_val)
  joblib.dump(scaler, scaler_path)
  ```
- **Confirmation**: `StandardScaler().fit_transform()` is called solely on `X_train`. Validation data (`X_val`) is normalized using the fitted training statistics via `.transform()`. Test data is not present in `train.py`.

#### (b) Data Loading per Augmentation Mode
- **File**: [`src/train.py`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/src/train.py#L71-L117)
- **Mode `none` (Lines 71–79)**:
  ```python
  if mode == "none":
      X = load_features(kind, n_coeff)
      tr_idx, va_idx, te_idx, y = load_split()
      meta = pd.read_csv(PROC_DIR / "splits.csv")
      X_train, y_train = X[tr_idx], y[tr_idx]
      X_val, y_val = X[va_idx], y[va_idx]
      meta_train = meta.iloc[tr_idx].copy()
      meta_val = meta.iloc[va_idx].copy()
  ```
  *Loads*: `features/emodb_{kind}_{n}.npy` (535 original clips) and `data/processed/splits.csv`. `X_train` receives 321 clips, `X_val` receives 107 clips.
- **Mode `after_split` (Lines 80–98)**:
  ```python
  elif mode == "after_split":
      feat_path = FEAT_DIR / f"version_a_{kind}_{n_coeff}.npy"
      label_path = FEAT_DIR / "version_a_labels.npy"
      meta_path = PROC_DIR / "version_a_metadata.csv"
      if not meta_path.is_file():
          meta_path = FEAT_DIR / "version_a_metadata.csv"

      X = np.load(feat_path)
      y = np.load(label_path)
      meta = pd.read_csv(meta_path)

      tr_idx = meta.index[meta["split"] == "train"].values
      va_idx = meta.index[meta["split"] == "val"].values

      X_train, y_train = X[tr_idx], y[tr_idx]
      X_val, y_val = X[va_idx], y[va_idx]
  ```
  *Loads*: `features/version_a_{kind}_{n}.npy` (1,819 rows) and `data/processed/version_a_metadata.csv`. `X_train` receives 1,605 augmented clips ($321 \times 5$). `X_val` receives 107 clean, unaugmented original clips (`set1`).
- **Mode `before_split` (Lines 99–117)**:
  ```python
  elif mode == "before_split":
      feat_path = FEAT_DIR / f"version_b_{kind}_{n_coeff}.npy"
      label_path = FEAT_DIR / "version_b_labels.npy"
      meta_path = PROC_DIR / "version_b_metadata.csv"
      if not meta_path.is_file():
          meta_path = FEAT_DIR / "version_b_metadata.csv"

      X = np.load(feat_path)
      y = np.load(label_path)
      meta = pd.read_csv(meta_path)

      tr_idx = meta.index[meta["split"] == "train"].values
      va_idx = meta.index[meta["split"] == "val"].values

      X_train, y_train = X[tr_idx], y[tr_idx]
      X_val, y_val = X[va_idx], y[va_idx]
  ```
  *Loads*: `features/version_b_{kind}_{n}.npy` (2,675 rows) and `data/processed/version_b_metadata.csv`. `X_train` receives 1,605 clips, `X_val` receives 535 clips.

---

### 3. Complete Codebase Search: Where Test Data is Touched

A full search of the codebase confirms that **test data is never accessed in training, early stopping, or checkpoint selection**:
- [`src/train.py`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/src/train.py): Lines 9, 10, 15, 138, 209 mention `test` only in explanatory comments explicitly stating that no test evaluation occurs. `te_idx` is unpacked from `load_split()` on line 73 but never sliced or referenced.
- [`src/model.py`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/src/model.py): Contains no test data references (only a dummy forward smoke test on random noise).
- [`scripts/run_grid.py`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/scripts/run_grid.py): Orchestrates `train_run` from `src/train.py`.
- **Files that legitimately touch test data**:
  1. [`src/make_splits.py`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/src/make_splits.py#L8-L11): Generates the original 60/20/20 partition `data/processed/splits.csv`.
  2. [`src/augment.py`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/src/augment.py#L274): Generates metadata splitting tags (`val`/`test` rows in Version A and B) and integrity assertion functions.
  3. [`src/test_augment.py`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/src/test_augment.py): Unit test assertions ensuring 0% split overlap for Version A and 535 rows in Version B test.
  4. [`scripts/compute_leakage.py`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/scripts/compute_leakage.py): Measures the percentage of test rows in Version B that share source clips with train.
  5. [`src/cache.py`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/src/cache.py#L29): Returns `test_idx` in `load_split()` for downstream evaluation.
  6. [`src/sanity.py`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/src/sanity.py#L25): Week 1 baseline SVM / Logistic Regression evaluation.
  7. [`src/evaluate.py`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/src/evaluate.py#L47-L84): Loads test data exclusively in `load_test_data()` and evaluates it in `evaluate_test_set()` *after* `selected_models.md` has been written and frozen.

---

### 4. Training Convergence and Overfitting Audit

Examining [`results/training_curves_best_runs.png`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/results/training_curves_best_runs.png) and the final training summaries for the best run in each setting:

| Feature | Coeffs ($n$) | Augmentation Mode | Seed | Final Train Acc | Final Val Acc | Final Train Loss | Final Val Loss | Overfitting Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **MFCC** | 12 | `none` | 42 | **100.0%** | 67.29% | 0.0018 | 1.8452 | **Severe Overfitting** (32.7% train-val gap; val loss diverges) |
| **MFCC** | 12 | `after_split` | 42 | **96.39%** | 72.90% | 0.1169 | 2.9602 | **Overfitting** (23.5% gap; high val loss) |
| **MFCC** | 12 | `before_split` | 42 | **98.57%** | 83.18% | 0.0391 | 0.9067 | Moderate gap (artificially depressed val loss via leakage) |
| **MFCC** | 24 | `none` | 43 | **100.0%** | 72.90% | 0.0001 | 1.6361 | **Severe Overfitting** (27.1% gap; train loss 0.0001) |
| **MFCC** | 24 | `after_split` | 42 | **96.51%** | 72.90% | 0.1052 | 1.3147 | **Overfitting** (23.6% gap) |
| **MFCC** | 24 | `before_split` | 43 | **99.88%** | 89.72% | 0.0039 | 0.5699 | High train/val accuracy (corrupted by leakage) |
| **MFCC** | 30 | `none` | 43 | **100.0%** | 72.90% | 0.0001 | 1.6210 | **Severe Overfitting** (27.1% gap; train loss 0.0001) |
| **MFCC** | 30 | `after_split` | 42 | **99.94%** | 73.83% | 0.0044 | 2.5352 | **Severe Overfitting** (26.1% gap; val loss reaches 2.54) |
| **MFCC** | 30 | `before_split` | 43 | **99.88%** | 91.40% | 0.0066 | 0.5786 | High train/val accuracy (corrupted by leakage) |
| **MFMC** | 12 | `none` | 43 | **99.07%** | 62.62% | 0.0543 | 1.7941 | **Severe Overfitting** (36.4% gap) |
| **MFMC** | 12 | `after_split` | 44 | **95.64%** | 73.83% | 0.1154 | 2.0563 | **Overfitting** (21.8% gap) |
| **MFMC** | 12 | `before_split` | 43 | **96.57%** | 77.57% | 0.0887 | 0.8880 | Moderate gap |
| **MFMC** | 24 | `none` | 42 | **98.75%** | 71.96% | 0.0326 | 2.3334 | **Severe Overfitting** (26.8% gap; val loss 2.33) |
| **MFMC** | 24 | `after_split` | 43 | **95.39%** | 77.57% | 0.1259 | 1.1866 | **Overfitting** (17.8% gap) |
| **MFMC** | 24 | `before_split` | 42 | **99.69%** | 85.61% | 0.0141 | 0.7038 | High train/val accuracy (corrupted by leakage) |
| **MFMC** | 30 | `none` | 43 | **100.0%** | 71.96% | 0.0011 | 2.0861 | **Severe Overfitting** (28.0% gap) |
| **MFMC** | 30 | `after_split` | 43 | **96.82%** | 74.77% | 0.1113 | 1.5905 | **Overfitting** (22.0% gap) |
| **MFMC** | 30 | `before_split` | 43 | **100.0%** | 88.04% | 0.0021 | 0.7039 | High train/val accuracy (corrupted by leakage) |

**Conclusion on Overfitting**:
**All models suffer from substantial overfitting.** In the unaugmented setup (`none`), the 1D-CNN fits the 321 training samples to 100% accuracy with near-zero loss by epoch 40, while validation loss climbs above 1.6–2.3. In `after_split`, train accuracy reaches 95–99% while validation accuracy plateaus in the 70–77% range with climbing validation loss.

---

### 5. Confusion Matrix Verification

- **Data Partition Used**: [`src/evaluate.py`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/src/evaluate.py#L313-L325) computes the confusion matrices directly from `y_true = np.array(best_run["y_true"])` and `y_pred = np.array(best_run["y_pred"])`, which are generated exclusively from the test set in `evaluate_test_set()` (lines 155–171).
- **Emotion Name Alignment**:
  - `EMOTION_NAMES = ["anger", "boredom", "disgust", "fear", "happiness", "neutral", "sadness"]`
  - In [`data/processed/metadata.csv`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/data/processed/metadata.csv), categorical encoding orders the strings alphabetically:
    `0: anger, 1: boredom, 2: disgust, 3: fear, 4: happiness, 5: neutral, 6: sadness`.
  - The matrix axes match the internal integer training classes exactly.

---

### 6. List of Suspicious Observations

The following anomalies are recorded directly without rationalizing:

1. **Large Standard Deviations on a Very Small Test Set**:
   - The test set has only 107 clips. Each clip represents approximately **0.93%** of the score.
   - Run-to-run variation across seeds produces standard deviations up to **±6.7%** (e.g., MFCC-12 `after_split` is $64.5 \pm 6.7\%$, where seed 43 got 70.09% and seed 44 got 57.01%—a 13-point swing on the same data).
2. **Substantial Baseline Gap Under No Augmentation**:
   - The paper reports **86.9%** without augmentation on EmoDB.
   - Our 1D-CNN without augmentation reaches only **64.2% to 71.0%** test accuracy. Data leakage cannot explain this gap because neither our `none` setup nor an honest no-augmentation setting contains augmented duplicates.
3. **Severe Divergence Between Training and Validation Loss**:
   - Training losses drop to between 0.0001 and 0.12 across all runs, while validation losses climb as high as 1.3 to 2.96. The 1D-CNN architecture overfits the pooled feature vectors regardless of Gaussian noise augmentation.
4. **Capacity Discrepancy Between 12 and 24/30 Coefficients**:
   - 12 coefficients yield a 64-unit flatten dimension into `Dense(120)` (49,687 total parameters), while 24 and 30 coefficients yield 192 units (65,047 parameters). The 12-coefficient model has a smaller classification head, so performance differences cannot be attributed solely to acoustic feature representations.
5. **Version B (before_split) Low Validation Loss**:
   - Version B exhibits validation losses of 0.56–0.91, in stark contrast to Version A (>1.2–2.9). This low loss reflects memorization of identical recordings appearing across splits.
