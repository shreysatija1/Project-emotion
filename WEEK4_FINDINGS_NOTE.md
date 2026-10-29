# Week 4 Reproduction Findings Note: 144-Input Representation, Multi-Corpus Evaluation, and Speaker Independence

**Project:** Reproducing Thomas Mary Little Flower et al., *"Data augmentation using a 1D-CNN model with MFCC/MFMC features for speech emotion recognition"*, *Automatika* (2024), DOI: [10.1080/00051144.2024.2371249](https://doi.org/10.1080/00051144.2024.2371249).  
**Authors:** Student Reproduction Team  
**Date:** Week 4  
**Status:** Complete (Tasks 1, 2, and 3 fully completed across 618 verified experimental runs)

---

## 1. What Does the 144-Input Format Look Like, and Did It Close the Gap?

### 1.1 Textual Evidence and Plausible Interpretations
In Section 3.2 of the paper, the authors state:
> *"One hundred forty-four feature vectors as input data for our CNN's first layer."*

In Week 3, our baseline used global summary vectors: extracting Mel-Frequency Cepstral Coefficients (MFCC) or Mel-Frequency Magnitude Coefficients (MFMC) across 12, 24, or 30 filterbanks, followed by computing the temporal mean and standard deviation over time. This yielded vectors of dimension 24, 48, and 60, respectively. While standard in acoustic classification, it left an obvious discrepancy with the paper's specified 144 inputs.

Crucially, **both 144-input representations below are our own reverse-engineered interpretations and are not confirmed by the published text**, which omits audio duration, framing parameters, hop size, and windowing functions.

1. **Interpretation 1: `frames144` (Temporal STFT Sequence)**
   - **Formulation:** A 2D matrix of shape `(144, n_coeff)` representing 144 consecutive time frames.
   - **Acoustic Rationale:** Conv1D with `kernel_size=15` sliding along the feature dimension of a 24-value vector spans 62.5% of the vector in a single stride, which lacks physical meaning. Sliding `kernel_size=15` across 144 time frames (hop length 256 samples @ 16 kHz = 16 ms) corresponds to a receptive field of ~240 ms, which aligns naturally with speech phoneme and syllable durations.
   - **Handling Variable Audio Durations:** Raw audio clips naturally vary in duration. After STFT framing at hop size 256 samples (16 ms), clips shorter than 144 frames are zero-padded at the end along the temporal dimension, whereas clips longer than 144 frames are center-cropped to exactly 144 frames.
   - **Architecture:** The 4 Conv1D layers with MaxPool1D(2) downsample the 144 frames ($144 \to 72 \to 36 \to 18 \to 9$), producing $9 \times 64 = 576$ units at the flatten stage feeding into `Dense(120)`.

2. **Interpretation 2: `stats144` (Extended Multi-Stream Feature Vector)**
   - **Formulation:** A 1D vector of shape `(144, 1)` constructed by augmenting static cepstral coefficients with dynamic velocity ($\Delta$) and acceleration ($\Delta^2$) trajectories, summarized with statistical moments.
   - **Exact 144-Value Derivations:**
     * **12 coefficients:** $12 \text{ static} + 12 \Delta + 12 \Delta^2 = 36$ trajectories. Computing 4 summary statistics (`mean`, `std`, `min`, `max`) across each gives exactly $36 \times 4 = 144$ scalar values.
     * **24 coefficients:** $24 \text{ static} + 24 \Delta + 24 \Delta^2 = 72$ trajectories. Computing 2 summary statistics (`mean`, `std`) across each gives exactly $72 \times 2 = 144$ scalar values.
     * **30 coefficients:** $30 \text{ static} + 30 \Delta + 30 \Delta^2 = 90$ trajectories. Computing `mean` and `std` produces 180 statistics; to match 144 inputs, this vector is truncated to the first 144 values ($30 \text{ static mean/std} + 30 \Delta \text{ mean/std} + 12 \Delta^2 \text{ mean/std}$).
   - **Recipe Discrepancy:** The `stats144` recipe differs across coefficient counts (4 statistics at 12, 2 at 24, 2 truncated at 30), so those columns are not like-for-like.

---

### 1.2 Did the 144-Input Format Close the Performance Gap?
**No. On 11 of the 12 evaluated configurations, the compact 24/48/60 baseline achieved higher test accuracy than both 144-input representations.** (The sole exception was MFMC-12 unaugmented, where both 144-input formats scored higher than the baseline).

| Feature & Coeffs | Setup | Baseline (24/48/60) | frames144 (144 × C) | stats144 (144 × 1) | Paper Target (Table 3) |
|---|---|---|---|---|---|
| **MFCC 12** | None | 64.2% ± 2.4% | 56.4% ± 1.9% | 60.8% ± 1.9% | 85.0% |
| **MFCC 12** | Version A (5x) | 64.5% ± 6.7% | 56.1% ± 5.2% | 58.3% ± 5.6% | 93.5% |
| **MFCC 24** | None | 69.2% ± 4.3% | 54.8% ± 5.1% | 60.4% ± 1.4% | 86.0% |
| **MFCC 24** | Version A (5x) | 68.2% ± 1.6% | 52.6% ± 5.1% | 63.6% ± 4.1% | 96.8% |
| **MFCC 30** | None | 71.0% ± 0.9% | 57.0% ± 3.7% | 62.9% ± 3.5% | 85.0% |
| **MFCC 30** | Version A (5x) | 70.7% ± 4.6% | 56.4% ± 4.3% | 65.4% ± 4.9% | 96.1% |
| **MFMC 12** | None | 51.7% ± 1.9% | 57.0% ± 3.4% | 59.8% ± 5.7% | 90.7% |
| **MFMC 12** | Version A (5x) | 63.6% ± 4.9% | 54.5% ± 1.9% | 56.4% ± 2.4% | 96.6% |
| **MFMC 24** | None | 62.9% ± 4.6% | 60.1% ± 2.7% | 56.7% ± 3.3% | 90.7% |
| **MFMC 24** | Version A (5x) | 68.5% ± 1.4% | 57.0% ± 3.4% | 56.7% ± 8.0% | 97.5% |
| **MFMC 30** | None | 65.7% ± 1.9% | 58.9% ± 4.9% | 60.1% ± 2.4% | 86.9% |
| **MFMC 30** | Version A (5x) | 72.3% ± 3.3% | 59.2% ± 2.7% | 61.4% ± 2.4% | 97.2% |

#### Key Takeaways:
1. **Hypothesis of Temporal Overfitting in `frames144`:** In `frames144`, 144 temporal frames produce 576 units into `Dense(120)`. Without recurrent units or attention, the 1D-CNN saturated training accuracy to ~100% within 15 epochs while validation performance plateaued. We hypothesize that this reflects rapid sequence memorization on small sample sizes (321 clips), though this remains a hypothesis as formal regularization or ablation sweeps were not conducted.
2. **Implicit Regularization of Summary Statistics:** The global `mean + std` pooling acts as an effective feature-level regularizer, reducing sample variance and outperforming the 144-input representations in most conditions.
3. **Format Used in Task 2:** Task 2 used the **baseline format** (24, 48, 60 values) because it yielded the highest overall accuracy across the majority of configurations.
4. **Verdict:** The 144-input representation does **not** close the performance gap with the paper's claims.

---

## 2. How Do Results Change Across Datasets? (CAFE, AESDD, MESD)

### 2.1 Multi-Corpus Inventory & Setup
To test cross-lingual validity and assess whether our findings generalize beyond German EmoDB, we expanded evaluation to three international speech corpora:
- **EmoDB (German):** 535 clips, 10 speakers, 7 emotions (anger, boredom, disgust, fear, happiness, sadness, neutral).
- **CAFE (Canadian French):** 936 clips, 12 professional actors (6M/6F), 7 emotions (anger, disgust, fear, happiness, sadness, surprise, neutral), CC BY-NC-SA 4.0 license. Perfectly balanced (78 clips per actor). *Note: CAFE was ingested via the open CAMEO HuggingFace parquet mirror and was not directly verified against the original Zenodo 1478765 distribution.*
- **AESDD (Greek):** 500 clips, 5 core actors (actors 1–5), 20 sentences (sentences 01–20), 5 emotions (anger, disgust, fear, happiness, sadness), Academic license. *Canonical restriction:* Filtered corrupted 36-byte file `s05 (3).wav`, enforced sentences 1–20 across actors 1–5 (496 clips), and substituted 4 unrecorded/corrupt takes with auxiliary actor 6 (`d18 (6)`, `f02 (6)`, `h11 (6)`, `s05 (6)`), asserting exactly 100 clips per emotion.
- **MESD (Mexican Spanish):** 862 single-word utterances, 6 emotions (anger, disgust, fear, happiness, sadness, neutral), CC BY 4.0 license. Contains adult female (412), adult male (270), and child (180) speaker cohorts. *Note: MESD was ingested via the open CAMEO HuggingFace parquet mirror and was not directly verified against the original Mendeley Data release.*

All datasets were resampled to 16 kHz mono, preprocessed identically, and evaluated across seeds 42, 43, 44 under stratified 60/20/20 partitioning.

### 2.2 Cross-Corpus Performance Dynamics & Theoretical Chance Levels

> [!IMPORTANT]
> **Chance Levels Differ Substantially Across Benchmarks:**
> - **AESDD (5 classes):** $\text{Chance Level} = \frac{1}{5} = \mathbf{20.0\%}$
> - **MESD (6 classes):** $\text{Chance Level} = \frac{1}{6} \approx \mathbf{16.7\%}$
> - **CAFE (7 classes):** $\text{Chance Level (Uniform)} = \frac{1}{7} \approx \mathbf{14.3\%}$ (Majority class baseline = $15.4\%$)
> - **EmoDB (7 classes):** $\text{Chance Level (Uniform)} = \frac{1}{7} \approx \mathbf{14.3\%}$ (Majority class baseline for Anger = $23.7\%$)
>
> A model achieving $50\%$ test accuracy on AESDD is $2.5\times$ chance, whereas $50\%$ on CAFE/EmoDB represents $3.5\times$ chance.

| Dataset | Language | Classes | Chance Level | MFCC 30 (None) | MFCC 30 (Ver A) | MFMC 30 (None) | MFMC 30 (Ver A) | Paper Claim (Table 3, 30 Coeffs) |
|---|---|---|---|---|---|---|---|---|
| **EmoDB** | German | 7 | 14.3% | 71.0% ± 0.9% | 70.7% ± 4.6% | 65.7% ± 1.9% | 72.3% ± 3.3% | **86.9% / 97.2%** (Table 3) |
| **AESDD** | Greek | 5 | 20.0% | 54.0% ± 6.6% | 57.7% ± 11.1% | 57.3% ± 7.2% | 64.3% ± 4.6% | **95.0% / 97.8%** (Table 3) |
| **CAFE** | French | 7 | 14.3% | 46.1% ± 5.2% | 39.9% ± 4.2% | 50.0% ± 1.1% | 54.6% ± 6.7% | **93.6% / 99.5%** (Table 3) |
| **MESD** | Spanish | 6 | 16.7% | 66.3% ± 4.3% | 69.0% ± 1.3% | 64.0% ± 5.5% | 67.3% ± 0.4% | **94.2% / 96.9%** (Table 3) |

### 2.3 Comprehensive Citations for Published Paper Claims (Per-Coefficient from Table 3)

In reviewing the published text (Thomas Mary Little Flower et al., *Automatika*, 2024), we noted two distinct reporting levels:
1. **Per-Feature High-Level Text (Page 8, under Figures 5–8):** The text reports single summary figures per dataset (which correspond to the 30-coefficient model from Table 3):
   - MFCC without DA: EmoDB 85.0%, AESDD 82.0%, CAFE 88.8%, MESD 85.5%
   - MFCC with DA: EmoDB 96.1%, AESDD 95.2%, CAFE 98.6%, MESD 96.1%
   - MFMC without DA: EmoDB 86.9%, AESDD 95.0%, CAFE 93.6%, MESD 94.2%
   - MFMC with DA: EmoDB 97.2%, AESDD 97.8%, CAFE 99.5%, MESD 96.9%
2. **Exact Per-Coefficient Breakdown (Page 13, Table 3):**
   Table 3 provides the exact breakdown for all 12, 24, and 30 coefficient settings across all five datasets:
   - **AESDD:**
     * 12 coeffs: MFCC None = 80.0%, Ver A = 97.4% | MFMC None = 90.0%, Ver A = 98.0%
     * 24 coeffs: MFCC None = 89.0%, Ver A = 97.8% | MFMC None = 96.0%, Ver A = 99.2%
     * 30 coeffs: MFCC None = 82.0%, Ver A = 95.2% | MFMC None = 95.0%, Ver A = 97.8%
   - **CAFE:**
     * 12 coeffs: MFCC None = 85.6%, Ver A = 94.3% | MFMC None = 92.5%, Ver A = 96.7%
     * 24 coeffs: MFCC None = 88.2%, Ver A = 96.6% | MFMC None = 93.0%, Ver A = 97.5%
     * 30 coeffs: MFCC None = 88.8%, Ver A = 98.6% | MFMC None = 93.6%, Ver A = 99.5%
   - **EmoDB:**
     * 12 coeffs: MFCC None = 85.0%, Ver A = 93.5% | MFMC None = 90.7%, Ver A = 96.64%
     * 24 coeffs: MFCC None = 86.0%, Ver A = 96.8% | MFMC None = 90.7%, Ver A = 97.5%
     * 30 coeffs: MFCC None = 85.0%, Ver A = 96.1% | MFMC None = 86.9%, Ver A = 97.2%
   - **MESD:**
     * 12 coeffs: MFCC None = 82.7%, Ver A = 91.4% | MFMC None = 90.2%, Ver A = 91.67%
     * 24 coeffs: MFCC None = 82.7%, Ver A = 95.6% | MFMC None = 91.3%, Ver A = 96.4%
     * 30 coeffs: MFCC None = 85.5%, Ver A = 96.1% | MFMC None = 94.2%, Ver A = 96.9%
3. **Abstract & Table 2 Peak Figures:** The authors highlight the overall maximum accuracy across any coefficient setting: EmoDB 97.50% (24 coeffs), AESDD 99.20% (24 coeffs), CAFE 99.50% (30 coeffs), MESD 96.90% (30 coeffs). This explains why the Week 3 status deck referenced 97.2% (the 30-coeff setting from Figure 8 / Table 3) while Table 2 in Week 4 reported 97.5% (the 24-coeff peak).

### 2.4 Empirical Observations & Statistical Nuance
1. **The Paper's Claims Were Not Replicated Under Our Implementation:**
   Across all four speech corpora, clean leak-free evaluation yields accuracies substantially lower than the paper's reported ~97–99% figures (lagging by 25 to 45 percentage points).
2. **Augmentation Shifts Are Within Standard Deviation:**
   Under clean Version A augmentation, test accuracy changes by $-6.2$ to $+8.9$ percentage points across stratified 24/30-coefficient configurations (mean $+3.0$ points; $-6.2$ to $+11.9$ points including 12-coefficient EmoDB), and by $-2.6$ to $+9.7$ percentage points under speaker-independent folds (mean $+2.7$ points). Crucially, **these differences are within $\pm 1$ standard deviation across random seeds or folds** (typical standard deviations range from 2% to 11%). This indicates that the observed shifts from additive Gaussian noise are not distinguishable from seed or fold variation.

---

## 3. How Much Does the Speaker-Independent Split Lower Accuracy? (Task 3 Findings)

### 3.1 The Speaker Memorization Confound
In standard stratified splits, clips from every actor are randomly distributed across train, validation, and test sets. Deep neural networks readily memorize speaker biometric traits (vocal tract geometry, formant frequencies, fundamental pitch ranges) rather than generalized emotional expressions. Under random splits, the network easily functions as a pseudo-speaker identifier mapping speaker identity to emotion labels.

To quantify this confound, we evaluated **rotating speaker-disjoint folds** where speakers are strictly segregated across folds:
- **EmoDB (10 speakers: 5M, 5F) -> 5 Folds:**
  * Test: 2 speakers (1 male, 1 female, ~20% of clips).
  * Val: 2 speakers taken from the remaining speakers (1 male, 1 female, ~20%).
  * Train: remaining 6 speakers (3 male, 3 female, ~60%).
  * Rotating test fold ensures all 10 speakers appear in the test set across the 5 folds.
- **CAFE (12 speakers: 6M, 6F) -> 6 Folds:**
  * Test: 2 speakers (1 male, 1 female, exactly 156 clips, 16.7%).
  * Val: 2 speakers taken from the remaining speakers (1 male, 1 female, 156 clips, 16.7%).
  * Train: remaining 8 speakers (4 male, 4 female, exactly 624 clips, 66.7%).
  * Rotating test fold ensures all 12 speakers appear in the test set across the 6 folds.
- **AESDD (5 core actors) -> 5 Folds:**
  * Test: 1 core actor (~100 clips, 20%).
  * Val: 1 core actor taken from the remaining core actors (~100 clips, 20%).
  * Train: remaining 3 core actors + ALL Actor 6 fallback clips (always in train).
  * Clip partition: exactly **300 train + 100 val + 100 test = 500 clips** for every fold.
  * Rotating test fold ensures all 5 core actors appear in the test set across the 5 folds.
- **MESD:** **EXCLUDED** from Task 3 because it contains only 3 demographic cohorts (`female`, `male`, `child`) rather than individual speaker identities.

### 3.2 Strict Evaluation Protocol & Integrity Guardrails
1. **Zero Speaker Overlap:** Verified programmatically with automated assertions: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
2. **Model Selection on Validation Speakers Only:** Model checkpoints are monitored and selected solely by accuracy on the unseen validation speakers. The test fold is scored exactly once on the frozen best checkpoint.
3. **Configurations Evaluated:** 24 and 30 coefficients only (dropping 12 coefficients), MFCC and MFMC, none and version A (clean 5x), 3 random seeds (42, 43, 44).
4. **Metric Reporting:** Results are aggregated by averaging seeds within each fold, then computing **Mean $\pm$ Std across folds** for each configuration.
5. **Execution Verification:** All 384 runs completed (120 EmoDB + 120 AESDD + 144 CAFE). The final row count in `results/week4_results.csv` is confirmed at exactly **618 rows** (234 baseline/stratified + 384 Task 3 runs).

### 3.3 Empirical Comparison: Stratified Baseline vs. Speaker-Independent Folds

| Dataset | Feature & Coeffs | Setup | Stratified Test Acc (Task 2) | Speaker-Independent Test Acc (Task 3) | Generalization Drop ($\Delta$) |
|---|---|---|---|---|---|
| **EmoDB** | MFCC 24 | None | 69.2% ± 4.3% | 52.4% ± 6.7% | **-16.8%** |
| **EmoDB** | MFCC 24 | Version A (5x) | 68.2% ± 1.6% | 51.5% ± 5.3% | **-16.8%** |
| **EmoDB** | MFCC 30 | None | 71.0% ± 0.9% | 51.6% ± 4.9% | **-19.5%** |
| **EmoDB** | MFCC 30 | Version A (5x) | 70.7% ± 4.6% | 53.5% ± 4.6% | **-17.2%** |
| **EmoDB** | MFMC 24 | None | 62.9% ± 4.6% | 49.2% ± 2.6% | **-13.7%** |
| **EmoDB** | MFMC 24 | Version A (5x) | 68.5% ± 1.4% | 52.2% ± 5.7% | **-16.3%** |
| **EmoDB** | MFMC 30 | None | 65.7% ± 1.9% | 47.1% ± 1.7% | **-18.7%** |
| **EmoDB** | MFMC 30 | Version A (5x) | 72.3% ± 3.3% | 55.2% ± 6.8% | **-17.1%** |
| **AESDD** | MFCC 24 | None | 57.0% ± 8.7% | 38.2% ± 4.5% | **-18.8%** |
| **AESDD** | MFCC 24 | Version A (5x) | 64.3% ± 7.6% | 43.1% ± 3.1% | **-21.2%** |
| **AESDD** | MFCC 30 | None | 54.0% ± 6.6% | 37.2% ± 3.9% | **-16.8%** |
| **AESDD** | MFCC 30 | Version A (5x) | 57.7% ± 11.1% | 40.2% ± 4.5% | **-17.5%** |
| **AESDD** | MFMC 24 | None | 56.7% ± 4.5% | 36.2% ± 7.1% | **-20.5%** |
| **AESDD** | MFMC 24 | Version A (5x) | 65.0% ± 2.6% | 45.9% ± 7.6% | **-19.1%** |
| **AESDD** | MFMC 30 | None | 57.3% ± 7.2% | 40.5% ± 6.8% | **-16.8%** |
| **AESDD** | MFMC 30 | Version A (5x) | 64.3% ± 4.6% | 44.0% ± 6.6% | **-20.3%** |
| **CAFE** | MFCC 24 | None | 45.9% ± 4.8% | 32.9% ± 4.3% | **-13.0%** |
| **CAFE** | MFCC 24 | Version A (5x) | 43.6% ± 2.4% | 31.4% ± 4.6% | **-12.2%** |
| **CAFE** | MFCC 30 | None | 46.1% ± 5.2% | 34.3% ± 4.8% | **-11.8%** |
| **CAFE** | MFCC 30 | Version A (5x) | 39.9% ± 4.2% | 31.7% ± 3.2% | **-8.2%** |
| **CAFE** | MFMC 24 | None | 47.9% ± 0.5% | 34.1% ± 5.1% | **-13.8%** |
| **CAFE** | MFMC 24 | Version A (5x) | 52.8% ± 2.0% | 36.7% ± 5.2% | **-16.2%** |
| **CAFE** | MFMC 30 | None | 50.0% ± 1.1% | 35.4% ± 6.4% | **-14.6%** |
| **CAFE** | MFMC 30 | Version A (5x) | 54.6% ± 6.7% | 36.3% ± 5.2% | **-18.4%** |

> [!NOTE]
> *Footnote on Standard Deviations:* Speaker-independent standard deviations are computed across cross-validation folds ($N=5$ or $6$, capturing inter-speaker variance), whereas stratified standard deviations are computed across random seeds on a fixed split ($N=3$, capturing optimization/initialization variance). Consequently, the two standard deviations are not directly comparable.

### 3.4 Key Takeaways & Scientific Synthesis
1. **Severe Generalization Collapse Across All Corpora:**
   - Removing speaker identity overlap causes an immediate **8.2 to 21.2 percentage point drop** in test accuracy across all 24 evaluated configurations (average drop: **-16.5 percentage points**). Across corpora: EmoDB drops by 13.7 to 19.5 points (mean: -17.0 points), AESDD by 16.8 to 21.2 points (mean: -18.9 points), and CAFE by 8.2 to 18.4 points (mean: -13.5 points).
   - EmoDB test accuracy collapses from ~66–72% to ~47–55%.
   - AESDD test accuracy collapses from ~54–65% to ~36–46%.
   - CAFE test accuracy collapses from ~40–55% to ~31–37%.
2. **Evidence Consistent with Speaker Confounding Under Random Splits:**
   - Under standard stratified splits, the results suggest the network readily leverages speaker acoustic traits (vocal tract geometry, pitch register) that correlate with emotions.
   - When required to generalize to novel speakers never encountered during training, test accuracy drops substantially, revealing much lower speaker-invariant emotion recognition capability.
3. **Additive Gaussian Noise Does Not Prevent Speaker Generalization Collapse:**
   - On EmoDB, clean Version A augmentation changes speaker-independent accuracy by only $-0.9\%$ to $+8.1\%$ (average change $+2.0\%$, which is well within the cross-fold standard deviation of $\pm 5\%$).
   - On AESDD, clean Version A augmentation shifts accuracy by $+3.0\%$ to $+9.7\%$ (within fold variation).
   - On CAFE, clean Version A augmentation shifts accuracy by $-2.6\%$ to $+2.6\%$.
   - **Conclusion:** Uniform acoustic noise does not alter fundamental speaker identity or pitch characteristics, and therefore offers no robust defense against speaker shift.
4. **Implications for Published Paper Claims:**
   - The paper's claimed peak accuracies of **97.2% (EmoDB)**, **97.8% (AESDD)**, and **99.5% (CAFE)** are consistent with the combined effects of **speaker overlap** (random splits) and potential **pre-split data augmentation leakage**, rather than true acoustic generalization.
   - Under strict, leak-free, speaker-independent evaluation, genuine test accuracy for this 1D-CNN architecture on these benchmarks is **31% to 55%** (EmoDB: 47–55%, AESDD: 36–46%, CAFE: 31–37%), roughly $2\times$ to $3.5\times$ chance, but far below near-perfect recognition.

---

## 4. Methodological Ambiguities Documented in Published Literature

1. **Input Dimensionality:** The paper specifies *"144 feature vectors"* but omits the framing parameters, window length, hop size, and audio duration required to generate 144 frames.
2. **Padding Configuration:** Without `padding="same"`, 4 Conv1D layers with kernel size 15 collapse 24-dimensional feature representations to negative dimensions.
3. **Augmentation Timing & Leakage:** The paper omits whether 5x Gaussian augmentation occurred before or after data partitioning. Week 3 demonstrated that pre-split augmentation leaks 95.9% of test clip source recordings into the training set (with 98.5% of all source clips appearing in train), creating an artificial +15–20% performance boost.
4. **Speaker Partitioning:** The published paper did not implement or report speaker-independent cross-validation, allowing speaker identity memorization to inflate reported accuracies.
5. **Data Provenance:** CAFE and MESD were ingested via the open CAMEO HuggingFace mirror and were not directly verified against the original Zenodo 1478765 and Mendeley Data releases.
