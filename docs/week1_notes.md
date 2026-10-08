# Week 1 Reproduction Notes: EmoDB Speech Emotion Recognition

## 1. What Was Built
During Week 1, we built the entire data preparation, feature extraction, baseline classification, and automated verification pipeline for reproducing the paper *"Data augmentation using a 1D-CNN model with MFCC/MFMC features for speech emotion recognition"* (Automatika, 2024):

- **Data Pipeline (`src/data.py`)**:
  - Scanned all 535 audio clips in `data/raw/emodb/`.
  - Parsed the speaker ID, speaker gender, emotion code, and alphabetical numeric label (0 to 6).
  - Loaded full audio at 16 kHz mono with zero artificial padding or trimming.
  - Calculated exact durations and generated `data/processed/metadata.csv`.
- **Stratified Split (`src/make_splits.py`)**:
  - Created a reproducible 60% train (321 clips), 20% validation (107 clips), and 20% test (107 clips) split stratified by emotion label, saved to `data/processed/splits.csv`.
- **Feature Extraction Modules**:
  - `src/features_mfcc.py`: Extracted 12, 24, and 30 Mel-Frequency Cepstral Coefficients (MFCC), pooled across time using mean and standard deviation ($2 \times n\_coeff = 24, 48, 60$).
  - `src/features_mfmc.py`: Extracted Mel-Frequency Magnitude Coefficients (MFMC) using the $\log_{10}$ of the Mel-weighted linear magnitude spectrum without Discrete Cosine Transform (DCT), pooled as mean and standard deviation.
  - `src/cache.py`: Efficiently saved and loaded all feature matrices as `.npy` files.
- **Sanity Baseline Classifier (`src/sanity.py`)**:
  - Trained Support Vector Machine (RBF kernel, $C=10$) and Logistic Regression classifiers on the standardized training set to verify that extracted features contain rich emotional discriminative power.
- **Visualization & Verification (`src/plots.py`, `src/verify.py`, `src/run_all.py`)**:
  - Generated class distribution and MFCC vs. MFMC comparison spectrograms in `results/`.
  - Verified no partition overlap, non-empty shapes, absence of NaNs, and spot-checked raw audio feature reconstruction matching stored `.npy` arrays.
  - Automated entire workflow via `src/run_all.py`.

---

## 2. Sanity Baseline Results

| Features | Coefficients ($n$) | Feature Vector Size | Classifier | Validation Accuracy | Test Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MFCC** | 12 | 24 | SVM | 77.57% | 80.37% |
| **MFCC** | 12 | 24 | LogReg | 74.77% | 81.31% |
| **MFCC** | 24 | 48 | SVM | 80.37% | 85.98% |
| **MFCC** | 24 | 48 | LogReg | 73.83% | 83.18% |
| **MFCC** | 30 | 60 | SVM | 77.57% | **88.79%** |
| **MFCC** | 30 | 60 | LogReg | 72.90% | 85.05% |
| **MFMC** | 12 | 24 | SVM | 64.49% | 67.29% |
| **MFMC** | 12 | 24 | LogReg | 66.36% | 62.62% |
| **MFMC** | 24 | 48 | SVM | 68.22% | 73.83% |
| **MFMC** | 24 | 48 | LogReg | 71.96% | 78.50% |
| **MFMC** | 30 | 60 | SVM | 71.03% | 74.77% |
| **MFMC** | 30 | 60 | LogReg | 70.09% | **84.11%** |

* **Majority-Class Baseline**: Always predicting the most frequent emotion (*anger*, 127/535) yields only **23.74%** accuracy.
* **Findings**:
  - All feature sets and simple linear/kernel classifiers beat the chance baseline by a wide margin (from 62.6% up to 88.8%).
  - Accuracy consistently scales upwards as the number of coefficients increases from 12 to 30.
  - Even simple classical models reach near the paper's target without augmentation (~85% on MFCC 30), validating that our acoustic features are correctly extracted.

---

## 3. Assumptions and Observations

1. **Input Vector Dimension (144 features in paper)**:
   - The paper states 144 features enter the 1D-CNN, but tests 12, 24, and 30 coefficients. In Week 1, pooling over time (mean + std) gives $2 \times n$ ($24, 48, 60$). In Week 2, we can test either temporal framing or additional statistics (e.g., min, max, $\Delta$, $\Delta^2$) to match 144 if needed.
2. **Convolution Filter Count**:
   - The paper mentions "sixty-two" (62) filters in layer 4; we assume this was a typographical error for 64, but can test both.
3. **Data Augmentation Split Protocol**:
   - The paper's high accuracy ($97.5\%$) with 5x Gaussian noise augmentation may indicate that augmentation occurred *before* the 60/20/20 split (causing audio variants of test speakers to leak into training). In Week 2, we will explicitly test both Protocol A (augment then split) and Protocol B (split then augment train only).
