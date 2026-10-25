# Week 3 Reproduction Summary: 1D-CNN Speech Emotion Recognition

**Paper**: *"Data augmentation using a 1D-CNN model with MFCC/MFMC features for speech emotion recognition"*  
**Journal**: *Automatika* (2024), DOI: [10.1080/00051144.2024.2371249](https://doi.org/10.1080/00051144.2024.2371249)  
**Authors**: Thomas Mary Little Flower, Thirasama Jaya, and Sreedharan Christopher Ezhil Singh.

---

## 1. What Was Built & Executed

During Week 3, we completed the full deep learning experimental pipeline reproducing and evaluating the paper's 1D-CNN architecture and data augmentation claims on EmoDB:

1. **Setup & Leakage Analysis**:
   - Standardized canonical feature file naming to `version_a_*` and `version_b_*` ([`ASSUMPTIONS.md`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/ASSUMPTIONS.md)).
   - Conducted an audit of the before-split protocol ([`leakage_report.md`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/leakage_report.md)): in Version B (*augment-then-split*), **96.64%** of validation rows and **95.89%** of test rows contain audio variants of clips present in the training set.
   - Verified automated data tests ([`augment_report.md`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/augment_report.md)), including class proportions and zero-leakage constraints for Version A.

2. **1D-CNN Architecture ([`src/model.py`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/src/model.py))**:
   - Implemented 4 Conv1D blocks with filter progression (8, 16, 32, 64), kernel size 15, stride 1, `same` padding, Batch Normalization, ReLU, and MaxPooling1D(2).
   - Penultimate Dense layer with 120 units and Dropout (0.3 after conv4, 0.2 after dense).
   - Softmax classifier over 7 EmoDB emotion classes with Adam ($\eta=0.001$) and sparse categorical cross-entropy.

3. **Training & Experiment Grid ([`src/train.py`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/src/train.py), [`scripts/run_grid.py`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/scripts/run_grid.py))**:
   - Executed **54 training runs** across the complete matrix: 2 feature types (MFCC, MFMC) $\times$ 3 coefficient configurations (12, 24, 30) $\times$ 3 augmentation protocols (`none`, `after_split`, `before_split`) across 3 independent random seeds (42, 43, 44).
   - Implemented strict train-only standardization (`StandardScaler` fitted strictly on `X_train`).
   - Training strictly excluded test data. All 54 runs logged to [`PROGRESS.md`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/PROGRESS.md).

4. **Rigorous Two-Phase Evaluation ([`src/evaluate.py`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/src/evaluate.py))**:
   - Selected best checkpoints per configuration based solely on validation accuracy and froze [`selected_models.md`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/selected_models.md) *prior* to test set evaluation.
   - Scored test sets in a single frozen pass.
   - Generated final tables ([`final_results_table.csv`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/results/final_results_table.csv), [`final_results_table.png`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/results/final_results_table.png)), confusion matrices, and training curves.

---

## 2. Comprehensive Results Table

All entries report **Mean $\pm$ Standard Deviation** over seeds 42, 43, and 44:

| Feature Representation | Coefficients ($n$) | No Augmentation (`none`)<br>*(Train: 321 clips)* | Version A (`after_split`)<br>*(Train: 1,605 clips, Clean)* | Version B (`before_split`)<br>*(Train: 1,605 clips, Leaked)* |
| :--- | :---: | :---: | :---: | :---: |
| **MFCC** | 12 | Val: 67.0 ± 0.5%<br>Test: 64.2 ± 2.4% | Val: 71.3 ± 1.9%<br>Test: 64.5 ± 6.7% | Val: 82.5 ± 1.0%<br>Test: 79.8 ± 0.4% |
| **MFCC** | 24 | Val: 70.4 ± 2.9%<br>Test: 69.2 ± 4.3% | Val: 69.8 ± 2.7%<br>Test: 68.2 ± 1.6% | Val: 88.3 ± 1.7%<br>Test: 87.6 ± 2.3% |
| **MFCC** | 30 | Val: 71.0 ± 2.5%<br>Test: 71.0 ± 0.9% | Val: 71.6 ± 2.4%<br>Test: 70.7 ± 4.6% | Val: 90.3 ± 1.1%<br>Test: 89.3 ± 1.6% |
| **MFMC** | 12 | Val: 61.1 ± 1.4%<br>Test: 51.7 ± 1.9% | Val: 72.3 ± 1.4%<br>Test: 63.6 ± 4.9% | Val: 75.4 ± 1.9%<br>Test: 73.8 ± 1.3% |
| **MFMC** | 24 | Val: 68.2 ± 3.7%<br>Test: 62.9 ± 4.6% | Val: 76.3 ± 1.1%<br>Test: 68.5 ± 1.4% | Val: 85.2 ± 0.4%<br>Test: 82.8 ± 1.8% |
| **MFMC** | 30 | Val: 68.5 ± 3.3%<br>Test: 65.7 ± 1.9% | Val: 74.5 ± 0.5%<br>Test: 72.3 ± 3.3% | Val: 86.7 ± 1.6%<br>Test: 87.2 ± 1.1% |

---

## 3. Main Findings & Analysis

### 1. Does Data Augmentation Help?
- The evidence **suggests** that augmentation has a mixed effect depending on feature type and dimensionality.
- Because the test set contains only 107 clips (where a single clip represents approximately 0.93% of the score) and run-to-run standard deviations reach 4% to 7%, small shifts in test accuracy must be interpreted with caution.
- **Only the MFMC 12-coefficient gain (+11.9%, from 51.7 ± 1.9% to 63.6 ± 4.9%) clearly stands out** as a substantial difference.
- For MFMC at 24 and 30 coefficients, the data suggests modest potential improvements (+5.6% and +6.6%), but standard deviations overlap considerably.
- For MFCC, the data suggests that augmentation has little noticeable effect (+0.3% at 12 coeffs, -1.0% at 24 coeffs, -0.3% at 30 coeffs).
- As an **untested hypothesis**, it is conceivable that DCT decorrelation in MFCC leaves less room for additive white Gaussian noise to act as a beneficial regularizer compared to raw mel-filterbank magnitude spectra in MFMC, but this hypothesis was not directly evaluated in our experiments.

### 2. MFCC vs. MFMC
- Without augmentation (`none`), MFCC test accuracy exceeds MFMC across all coefficient sizes (e.g., at 30 coeffs: 71.0 ± 0.9% vs 65.7 ± 1.9%).
- With clean augmentation (`after_split`), MFMC test accuracy reaches parity with MFCC (72.3 ± 3.3% vs 70.7 ± 4.6% at 30 coeffs).

### 3. After-Split vs. Before-Split & The Gap to the Paper
- **Our leak-free result is far below the paper.** Under honest split-then-augment (Version A), our best test performance reaches 70.7 ± 4.6% (MFCC-30) and 72.3 ± 3.3% (MFMC-30), compared to the paper's reported **97.2%** on EmoDB with augmentation.
- When we augment before splitting (Version B), test accuracy rises sharply to **89.3 ± 1.6%** (MFCC-30) and **87.2 ± 1.1%** (MFMC-30).
- Therefore, **data leakage is a plausible contributing factor** to the high accuracy numbers reported in the literature.
- However, our setup also differs from the paper in other notable ways:
  - Input vector size: 24, 48, and 60 in our implementation vs. the paper's stated 144.
  - Conv1D layer 4 filter count: 64 in our setup vs. 62 stated in the paper.
  - Batch size (32) and exact noise scaling / SNR protocol.
- Furthermore, **we could not find a clear statement in the paper of whether data augmentation happened before or after the train/val/test split.**
- Crucially, **leakage cannot explain why our no-augmentation baseline (about 64–71% test accuracy) is already far below the paper's 86.9% no-augmentation figure.** Because neither our unaugmented setup nor an honest no-augmentation baseline contains augmented copies, other factors—such as input representation, architecture details, or optimization differences—must account for this baseline gap.

---

## 4. Comparison with the Paper's Numbers

| Setting | Paper Claimed (EmoDB) | Reproduced (Version A: Clean) | Reproduced (Version B: Augment-First) | Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **No Augmentation Baseline** | **86.9%** | **64.2% – 71.0%** | — | **Gap of ~16–23%**: Leakage cannot explain this difference since no augmentation is used. |
| **With Augmentation** | **97.2%** | **70.7% – 72.3%** | **87.2% – 89.3%** | **Gap of ~25–26% vs Clean**: When augmenting before splitting, accuracy rises sharply toward the upper 80s/low 90s, suggesting leakage is a plausible factor, alongside other architectural and parameter differences. |

---

## 5. Assumptions and Deviations from the Paper

1. **Input Feature Dimensionality**:
   - The paper mentions an input size of 144, but tests 12, 24, and 30 coefficients. Mean and std pooled over time naturally produce 24, 48, and 60 dimensions.
2. **Convolution Padding**:
   - The paper specifies 4 Conv1D layers with kernel size 15 and MaxPooling1D(2). With valid padding, the feature map vanishes by layer 3; `padding='same'` was required.
3. **Layer 4 Filter Count**:
   - The paper text states 62 filters; our implementation defaults to 64 (standard power of 2).
4. **Architectural Capacity Differences Across Coefficients**:
   - As documented in [`ASSUMPTIONS.md`](file:///C:/Users/Shrey/OneDrive/Documents/Project%20Emotion/ASSUMPTIONS.md), 4 pooling layers reduce length 24 to 1 ($1 \times 64 = \mathbf{64}$ units into `Dense(120)`), whereas lengths 48 and 60 reduce to 3 ($3 \times 64 = \mathbf{192}$ units into `Dense(120)`).
   - Because the 12-coefficient model ends with a smaller flatten layer and fewer parameters (49,687 vs 65,047), performance comparisons between 12 and 24/30 coefficients are not purely about acoustic feature representations.

---

## 6. Open Problems & Next Steps

1. **Investigating the No-Augmentation Baseline Gap**:
   - Explore alternative input representations (e.g., temporal framing without global pooling to match the paper's 144-dimensional input) to understand why baseline accuracy on EmoDB remains below 86.9%.
2. **Speaker-Independent Cross-Validation**:
   - Implement Leave-One-Speaker-Out (LOSO) cross-validation to assess true speaker-independent emotion generalization across EmoDB's 10 speakers.
3. **Cross-Corpus Validation in Week 4**:
   - Evaluate model and augmentation behavior across additional corpora: CAFE, AESDD, and MESD.
