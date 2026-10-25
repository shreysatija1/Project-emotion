# Presentation Slides Outline: Week 3 Reproduction Findings

---

### Slide 1: Title & Research Objective
- **Title**: *Reproduction and Evaluation of 1D-CNN Speech Emotion Recognition on EmoDB*
- **Paper Reference**: Thomas Mary Little Flower et al., *Automatika* (2024), DOI: 10.1080/00051144.2024.2371249
- **Core Research Questions**:
  1. What is the observable effect of additive Gaussian noise data augmentation on 1D-CNN SER models?
  2. How do Mel-Frequency Magnitude Coefficients (MFMC) compare with Mel-Frequency Cepstral Coefficients (MFCC)?
  3. Does the data splitting protocol (augmenting before vs. after partitioning) influence reported performance?

---

### Slide 2: Pipeline & Feature Representation
- **Acoustic Feature Extraction**:
  - MFCC and MFMC extracted across 12, 24, and 30 filterbank bands ($2 \times n$ pooled mean + standard deviation over time: 24, 48, 60 features).
  - MFMC computes $\log_{10}$ mel-filterbank magnitude spectrum without discrete cosine transform (DCT).
- **1D-CNN Architecture**:
  - 4 Conv1D blocks (8, 16, 32, 64 filters; kernel size 15, stride 1, `same` padding, BatchNorm, ReLU, MaxPool1D(2)).
  - Dropout(0.3) $\to$ Flatten $\to$ Dense(120, ReLU) $\to$ Dropout(0.2) $\to$ Dense(7, Softmax).
  - Adam optimizer ($\eta = 0.001$), batch size 32, 100 epochs, sparse categorical cross-entropy.
  - *Structural note*: 12 coefficients flatten to 64 units, while 24 and 30 flatten to 192 units.

---

### Slide 3: Two Augmentation Protocols
- **Augmentation Scheme**: 5 representations per audio clip (original $x$, zero-noise copy $x$, and Gaussian noise $\sigma \in \{0.01, 0.025, 0.05\}$).
- **Protocol Comparison**:
  - **Version A (After-Split / Clean)**:
    - Train set (321 clips) augmented $5\times \to 1,605$ samples.
    - Validation (107 clips) and Test (107 clips) kept strictly unaugmented (original clean audio).
    - **0.0%** overlap between training source clips and val/test clips.
  - **Version B (Before-Split / Leaked)**:
    - All 535 clips augmented $5\times \to 2,675$ samples.
    - Random stratified 60/20/20 split across the 2,675 samples ($1,605 / 535 / 535$).

---

### Slide 4: Data Leakage Analysis in Version B
- **Quantitative Overlap in Version B**:
  - **98.5%** of all EmoDB source clips (527 / 535) appear in the training partition.
  - **96.6%** of validation samples (517 / 535) and **95.9%** of test samples (513 / 535) are variants of recordings seen in training.
- **Interpretation**:
  - In Version B, the model is tested on audio variants of clips it has already trained on.
  - We could not find a clear statement in the published paper of whether augmentation occurred before or after the train/val/test split.

---

### Slide 5: Methodological Guardrails
- **Train-Only Standardization**:
  - `StandardScaler` fitted strictly on `X_train`. Validation and test sets normalized using training statistics.
- **Strict Separation of Model Selection & Test Scoring**:
  - Best model checkpoint per setting selected solely by `val_accuracy` and frozen in `selected_models.md`.
  - Test partition scored exactly once per selected model without tuning loops.
- **Statistical Reliability**:
  - Evaluated across random seeds (42, 43, 44) to report mean and standard deviation for all 18 configurations.

---

### Slide 6: Experimental Results
- **Comprehensive Grid (2 Features $\times$ 3 Sizes $\times$ 3 Modes, Mean $\pm$ Std across 3 Seeds)**:
  - *No Augmentation (`none`)*: MFCC reaches 64.2% to 71.0%; MFMC reaches 51.7% to 65.7%.
  - *Clean Augmentation (Version A)*: MFCC reaches 64.5% to 70.7%; MFMC reaches 63.6% to 72.3%.
  - *Augment-First (Version B)*: MFCC reaches 79.8% to 89.3%; MFMC reaches 73.8% to 87.2%.

---

### Slide 7: Findings & Observations
- **Effect of Augmentation**:
  - Evidence suggests a mixed effect. On a 107-clip test set (~0.9% per clip) with standard deviations of 4–7%, only the MFMC 12-coefficient gain (+11.9%, 51.7% $\to$ 63.6%) clearly stands out.
  - MFCC gains are within margin of error (+0.3% to -1.0%).
  - An untested hypothesis is that DCT decorrelation in MFCC leaves less room for additive Gaussian noise to act as a regularizer compared to raw mel filterbank spectra.
- **Comparing Version A and Version B**:
  - Our leak-free result is far below the paper's reported 97.2%. When we augment before splitting, accuracy rises sharply to 87–89%, so leakage is a plausible contributing factor.
  - However, our setup also differs from the paper in other ways (input size 24/48/60 vs. 144, 64 vs. 62 filters, batch size, noise scaling).
  - Crucially, leakage cannot explain why our no-augmentation baseline (64–71%) is far below the paper's 86.9% no-augmentation figure.

---

### Slide 8: Confusion Matrix & Class Error Dynamics
- **Patterns on EmoDB (Clean Version A)**:
  - High accuracy on *Anger* and *Happiness* (high energy / distinctive acoustic profiles).
  - Frequent mutual confusion between *Neutral*, *Boredom*, and *Sadness* (low energy / subdued prosody).
  - Confusion matrices for best MFCC and MFMC models illustrate these shared acoustic boundary challenges.

---

### Slide 9: Assumptions & Differences from the Paper
- **Input Dimension**: Paper states 144; we evaluate true pooled dimensions (24, 48, 60).
- **Conv1D Padding**: Used `padding='same'` to prevent sequence length collapsing to zero with kernel size 15 across 4 pooling layers.
- **Filter Count**: Defaulted to 64 filters in Layer 4 (paper states 62).
- **Capacity Asymmetry**: 12 coefficients flatten to 64 units, while 24 and 30 flatten to 192 units.

---

### Slide 10: Next Steps: Week 4 Multi-Corpus Evaluation
- **Cross-Corpus Expansion**:
  - Extend pipeline to CAFE (Canadian French), AESDD (Greek), and MESD (Mexican Spanish).
  - Assess whether findings on augmentation and feature representations generalize across languages and recording environments.
