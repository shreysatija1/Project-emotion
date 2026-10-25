# Reviewer Verification Checklist (Week 3)

This checklist provides exact file paths and line numbers so you can independently verify that all methodological rules and anti-leakage constraints were strictly adhered to.

---

### 1. Test Set Scored ONLY After `selected_models.md` Was Written & Frozen

- **No Test Data in Training**:
  - File: [`src/train.py`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/src/train.py#L170-L195)
  - Lines: `170-195`
  - Verification: `model.fit()` runs strictly with `validation_data=(X_val_3d, y_val)`. Test data is neither loaded nor evaluated anywhere in `src/train.py`.

- **Strict Two-Phase Evaluation Order**:
  - File: [`src/evaluate.py`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/src/evaluate.py#L324-L334)
  - Lines: `324-334`
  - Verification: 
    1. Line `328`: `write_and_freeze_selected_models(summaries)` selects best models using `val_accuracy` and writes `selected_models.md` to disk.
    2. Line `331`: `evaluate_test_set(summaries)` is called only after model selection has been saved and frozen. Test evaluation is executed exactly once per model.

---

### 2. Feature Scaler Fitted on Training Split ONLY

- **Training Normalization**:
  - File: [`src/train.py`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/src/train.py#L137-L141)
  - Lines: `137-141`
  - Code:
    ```python
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    joblib.dump(scaler, scaler_path)
    ```
  - Verification: `fit_transform` is applied exclusively to `X_train`. Validation data is normalized using the fitted training statistics via `transform()`.

- **Test Set Normalization**:
  - File: [`src/evaluate.py`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/src/evaluate.py#L159-L163)
  - Lines: `159-163`
  - Code:
    ```python
    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test)
    X_test_3d = prepare_input(X_test_scaled)
    ```
  - Verification: Test features are transformed strictly using the loaded training scaler, preventing any backward information leakage.

---

### 3. Data Loading Partitioning per Augmentation Mode

- **Training Loader**:
  - File: [`src/train.py`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/src/train.py#L48-L105)
  - `none` Mode (Lines `58-67`): Loads unaugmented features from `features/emodb_{kind}_{n}.npy` and partitions via Week 2 `splits.csv` (Train: 321, Val: 107).
  - `after_split` Mode (Lines `69-86`): Loads Version A artifacts (`features/version_a_{kind}_{n}.npy` and `version_a_metadata.csv`). Train contains 1,605 augmented clips ($321 \times 5$); Val contains 107 original unaugmented clips (`set1` only).
  - `before_split` Mode (Lines `88-105`): Loads Version B artifacts (`features/version_b_{kind}_{n}.npy` and `version_b_metadata.csv`). Train contains 1,605 clips; Val contains 535 clips.

- **Test Loader**:
  - File: [`src/evaluate.py`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/src/evaluate.py#L38-L82)
  - `none` Mode (Lines `44-48`): Evaluates on the 107 test clips from `splits.csv`.
  - `after_split` Mode (Lines `50-65`): Evaluates on the 107 test clips from `version_a_metadata.csv` (identical to original test set, `set1` only).
  - `before_split` Mode (Lines `67-82`): Evaluates on the 535 test clips from `version_b_metadata.csv`.
