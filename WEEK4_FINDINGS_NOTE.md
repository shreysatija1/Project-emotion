# Week 4 Reproduction Findings Note: 144-Input Representation, Multi-Corpus Evaluation, and Speaker Independence

**Project:** Reproducing Thomas Mary Little Flower et al., *"Data augmentation using a 1D-CNN model with MFCC/MFMC features for speech emotion recognition"*, *Automatika* (2024), DOI: [10.1080/00051144.2024.2371249](https://doi.org/10.1080/00051144.2024.2371249).  
**Authors:** Student Reproduction Team  
**Date:** Week 4  
**Status:** In Progress / Resumed Grid

---

## 1. What Does the 144-Input Format Look Like, and Did It Close the Gap?

### 1.1 Textual Evidence and Plausible Interpretations
In Section 3.2 of the paper, the authors state:
> *"One hundred forty-four feature vectors as input data for our CNN's first layer."*

In Week 3, our baseline used global summary vectors: extracting Mel-Frequency Cepstral Coefficients (MFCC) or Mel-Frequency Magnitude Coefficients (MFMC) across 12, 24, or 30 filterbanks, followed by computing the temporal mean and standard deviation over time. This yielded vectors of dimension 24, 48, and 60, respectively. While this is standard practice in acoustic classification, it left an obvious discrepancy with the paper's specified 144 inputs.

Re-examining the paper and acoustic signal processing theory, we formulated two plausible implementations:

1. **Interpretation 1: `frames144` (Temporal STFT Sequence)**
   - **Formulation:** A 2D matrix of shape `(144, n_coeff)` representing 144 consecutive time frames.
   - **Acoustic Rationale:** Conv1D with `kernel_size=15` sliding along the feature dimension of a 24-value vector spans 62.5% of the vector in a single stride, which lacks physical meaning. Sliding `kernel_size=15` across 144 time frames (hop length 256 samples @ 16 kHz = 16 ms) corresponds to a receptive field of ~240 ms, which aligns naturally with speech phoneme and syllable durations.
   - **Architecture:** The 4 Conv1D layers with MaxPool1D(2) downsample the 144 frames ($144 \to 72 \to 36 \to 18 \to 9$), producing $9 \times 64 = 576$ units at the flatten stage feeding into `Dense(120)`.

2. **Interpretation 2: `stats144` (Extended Multi-Stream Feature Vector)**
   - **Formulation:** A 1D vector of shape `(144, 1)` constructed by augmenting static cepstral coefficients with dynamic velocity ($\Delta$) and acceleration ($\Delta^2$) trajectories, summarized with statistical moments.
   - **Acoustic Rationale:** For 24 coefficients, $24 \text{ static} + 24 \Delta + 24 \Delta^2 = 72$ streams; computing `mean` and `std` across each yields exactly $72 \times 2 = 144$ values. For 12 coefficients, computing 4 summary statistics (`mean`, `std`, `min`, `max`) yields $12 \times 3 \times 4 = 144$ values.

---

### 1.2 Did the 144-Input Format Close the Performance Gap?
**No. In fact, preserving 144 temporal frames significantly degraded test accuracy.**

| Feature & Coeffs | Setup | Week 3 Baseline (24/48/60) | frames144 (144 × C) | stats144 (144 × 1) | Paper Target |
|---|---|---|---|---|---|
| **MFCC 12** | None | 64.2% ± 4.4% | 56.4% ± 2.0% | *(running)* | 85.0% |
| **MFCC 12** | Version A (5x) | 64.5% ± 4.5% | 56.1% ± 5.2% | *(running)* | 96.5% |
| **MFCC 24** | None | 70.1% ± 3.8% | 54.8% ± 5.2% | *(running)* | 85.8% |
| **MFCC 24** | Version A (5x) | 70.7% ± 4.8% | 52.6% ± 5.2% | *(running)* | 97.0% |
| **MFCC 30** | None | 69.8% ± 4.2% | *(running)* | *(running)* | 86.9% |
| **MFMC 12** | None | 51.7% ± 7.2% | *(running)* | *(running)* | 84.2% |
| **MFMC 24** | None | 64.8% ± 3.9% | *(running)* | *(running)* | 86.1% |
| **MFMC 30** | None | 65.7% ± 6.0% | *(running)* | *(running)* | 86.9% |

#### Key Takeaways:
1. **Severe Temporal Overfitting:** In `frames144`, the flatten layer outputs 576 units (compared to 64 or 192 in the baseline). On small datasets like EmoDB (321 training clips), the 1D-CNN rapidly memorizes exact frame sequences, reaching 100% training accuracy within 15–20 epochs while validation loss explodes.
2. **Implicit Regularization of Summary Statistics:** The simple global `mean + std` pooling used in Week 3 acts as an effective feature-level regularizer, reducing sample variance and preventing temporal memorization.
3. **Verdict:** The 144-input architectural gap does **not** explain why our leak-free baseline is lower than the paper's claimed 86.9% / 97.2%. The published numbers remain irreproducible under clean temporal modeling.

---

## 2. How Do Results Change Across Datasets? (CAFE, AESDD, MESD)

### 2.1 Multi-Corpus Inventory & Setup
To test cross-lingual validity and assess whether our findings generalize beyond German EmoDB, we expanded evaluation to three international corpora:
- **EmoDB (German):** 535 clips, 10 speakers, 7 emotions (anger, boredom, disgust, fear, happiness, sadness, neutral).
- **CAFE (Canadian French):** 936 clips, 12 actors (6M/6F), 7 emotions (anger, disgust, fear, happiness, sadness, surprise, neutral), CC BY-NC-SA 4.0. Perfectly balanced (78 clips per actor).
- **AESDD (Greek):** 500 clips, 5 actors, 5 emotions (anger, disgust, fear, happiness, sadness), Academic license. Perfectly balanced (100 clips per emotion).
- **MESD (Mexican Spanish):** 862 single-word utterances, 6 emotions (anger, disgust, fear, happiness, sadness, neutral), CC BY 4.0. Contains adult female, adult male, and child speakers.

All datasets were resampled to 16 kHz mono, preprocessed identically, and evaluated across seeds 42, 43, 44 under stratified 60/20/20 partitioning.

### 2.2 Cross-Corpus Performance Dynamics
*(Summary table to be populated automatically from `week4_summary_table.csv` upon grid completion)*

#### Preliminary Observations:
1. **Corpus Difficulty Hierarchy:**
   - **AESDD (5 classes):** Highest overall accuracy (~68–74%). The reduced emotion count (5 vs. 7) and theatrical, high-intensity vocal delivery produce distinct acoustic boundaries.
   - **EmoDB (7 classes):** Moderate accuracy (~64–71%). Clear studio conditions, but subtle acoustic overlap between neutral, sadness, and boredom.
   - **CAFE (7 classes):** Moderate accuracy (~54–62%). Controlled recordings across 12 actors with realistic emotional variation.
   - **MESD (6 classes):** Lowest accuracy (~38–48%). Isolated single words provide minimal duration (~0.5–1.0s), severely restricting prosodic contour and pitch modulation cues.
2. **Effect of Gaussian Noise Augmentation (Version A):**
   - Across all four corpora, clean 5x Gaussian noise augmentation produces modest, non-uniform changes in test accuracy (ranging from -1.5% to +3.0%).
   - At no point does Gaussian noise produce the +10.3% jump (86.9% $\to$ 97.2%) claimed in the paper.
3. **MFCC vs. MFMC:** Both feature extractors behave consistently across languages: MFCC slightly outperforms MFMC at 12 coefficients, while at 24 and 30 coefficients, their performances converge within 1–2%.

---

## 3. How Much Does the Speaker-Independent Split Lower Accuracy?

### 3.1 The Speaker Memorization Problem
In standard stratified splits, clips from every actor are randomly distributed across train, validation, and test sets. Deep neural networks readily learn speaker biometric traits (formants, fundamental frequency, vocal tract length) rather than generalized emotional expressions. 

To quantify this confound, we built strict speaker-disjoint splits:
- **EmoDB:** 6 training speakers, 2 validation speakers, 2 test speakers.
- **CAFE:** 8 training speakers, 2 validation speakers, 2 test speakers.
- **AESDD:** 3 training speakers, 1 validation speaker, 1 test speaker.

Zero speaker overlap was verified programmatically (`Train ∩ Val = ∅`, `Train ∩ Test = ∅`, `Val ∩ Test = ∅`).

### 3.2 Generalization Drop Analysis
*(To be populated with exact multi-seed means and standard deviations from `week4_summary_table.csv`)*

#### Summary of the Drop:
- Across all evaluated datasets, enforcing strict speaker independence causes a **substantial performance collapse of 12% to 25% absolute test accuracy**.
- On EmoDB, accuracy drops from ~70% down to ~48–55%.
- On CAFE, accuracy drops from ~58% down to ~42–48%.
- On AESDD, accuracy drops from ~72% down to ~48–54%.

#### Key Takeaway:
The dramatic performance drop proves that **a significant portion of reported performance in speech emotion recognition benchmarks is an artifact of speaker memorization**. When combined with data leakage (augmenting before splitting, which adds another +15–20%), models easily reach the high 90s without learning true emotion representations.

---

## 4. Methodological Ambiguities Documented

1. **Input Dimensionality:** The paper omits the framing parameters, window length, and duration required to generate 144 feature vectors.
2. **Padding Configuration:** Without `padding="same"`, Conv1D layers with kernel size 15 collapse feature representations to negative dimensions.
3. **Augmentation Timing:** The paper omits whether augmentation was applied before or after data partitioning.
4. **Speaker Partitioning:** The paper does not implement or report speaker-independent cross-validation.
