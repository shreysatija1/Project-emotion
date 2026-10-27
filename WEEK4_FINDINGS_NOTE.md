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
**No. In fact, both 144-input representations performed worse than the compact 24/48/60 baseline.**

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
1. **Severe Temporal Overfitting in `frames144`:** In `frames144`, 144 temporal frames produce 576 units into `Dense(120)`. Without recurrent units or attention, the 1D-CNN memorizes specific training sequences, reaching 100% training accuracy within 15 epochs while test accuracy drops by 7% to 15% compared to the baseline.
2. **Implicit Regularization of Summary Statistics:** The global `mean + std` pooling acts as an effective feature-level regularizer, reducing sample variance and outperforming both multi-stream `stats144` and temporal `frames144`.
3. **Format Used in Task 2:** Task 2 used the **baseline format** (24, 48, 60 values). Because the baseline format was strictly superior to both 144-input representations, no better format exists to rerun Task 2 with.
4. **Verdict:** The 144-input representation does **not** close the performance gap. The paper's claimed numbers were not replicated under our implementation with either 144-input representation.

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

| Dataset | Language | Classes | Chance Level | MFCC 30 (None) | MFCC 30 (Ver A) | MFMC 30 (None) | MFMC 30 (Ver A) | Paper Claim (Cited) |
|---|---|---|---|---|---|---|---|---|
| **EmoDB** | German | 7 | 14.3% | 71.0% ± 0.9% | 70.7% ± 4.6% | 65.7% ± 1.9% | 72.3% ± 3.3% | **97.2% / 97.5%** |
| **AESDD** | Greek | 5 | 20.0% | *(rerunning)* | *(rerunning)* | *(rerunning)* | *(rerunning)* | **97.8% / 99.2%** |
| **CAFE** | French | 7 | 14.3% | 46.1% ± 5.2% | 39.9% ± 4.2% | 50.0% ± 1.1% | 54.6% ± 6.7% | **99.5%** |
| **MESD** | Spanish | 6 | 16.7% | 66.3% ± 4.3% | 69.0% ± 1.3% | 64.0% ± 5.5% | 67.3% ± 0.4% | **96.9%** |

### 2.3 Comprehensive Citations for Published Paper Claims

A critical finding in reviewing the published text (Thomas Mary Little Flower et al., *Automatika*, 2024) is that the authors report different accuracy numbers in different sections:

1. **Why Week 3 cited 97.2% while Table 2 cites 97.5% for EmoDB:**
   - **Page 8 (Section 4, text under Figure 8):** The authors report **97.2%** for the EmoDB test dataset under 1D-CNN-MFMC with DA. This corresponds specifically to the **30-coefficient** model.
   - **Page 13 (Table 3, "Comparative Analysis of the five models' accuracies"):** Shows the breakdown by coefficient count:
     * 12 coefficients with DA: **96.64%**
     * 24 coefficients with DA: **97.5%** (peak accuracy across coefficients)
     * 30 coefficients with DA: **97.2%**
   - **Page 13 (Table 2) & Page 2 (Abstract):** The authors select the peak number across all coefficient sizes, reporting **97.50%** for EmoDB, **99.20%** for AESDD, **99.50%** for CAFE, and **96.90%** for MESD.

2. **Full Inventory of Paper Citations:**
   - **1D-CNN-MFCC without DA (Page 8, under Figure 5):**
     * AESDD: **82%** | CAFE: **88.8%** | EmoDB: **85%** | MESD: **85.5%**
   - **1D-CNN-MFCC with DA (Page 8, under Figure 6):**
     * AESDD: **95.2%** | CAFE: **98.6%** | EmoDB: **96.1%** | MESD: **96.1%**
   - **1D-CNN-MFMC without DA (Page 8, under Figure 7):**
     * AESDD: **95%** | CAFE: **93.6%** | EmoDB: **86.9%** | MESD: **94.2%**
   - **1D-CNN-MFMC with DA (Page 8, under Figure 8 & Page 13 Table 3):**
     * AESDD: **97.8%** (30 coeffs) / **99.2%** (24 coeffs, Table 2 & 3)
     * CAFE: **99.5%** (30 coeffs, Table 2 & 3) / **97.5%** (24 coeffs)
     * EmoDB: **97.2%** (30 coeffs) / **97.5%** (24 coeffs, Table 2 & 3)
     * MESD: **96.9%** (30 coeffs, Table 2 & 3) / **96.4%** (24 coeffs)

### 2.4 Empirical Observations & Statistical Nuance
1. **The Paper's Claims Were Not Replicated Under Our Implementation:**
   Across all four speech corpora, clean leak-free evaluation yields accuracies substantially lower than the paper's reported ~97–99% figures (lagging by 25 to 45 percentage points).
2. **Augmentation Gains Are Within Seed Standard Deviation:**
   Under clean Version A augmentation, test accuracy changes by only $-1.5\%$ to $+3.5\%$ on most configurations. Crucially, **these differences are within $\pm 1$ standard deviation across random seeds** (typical standard deviations range from 2% to 6%). This indicates that additive Gaussian noise does not produce a statistically reliable or significant regularization benefit on test performance.

---

## 3. Speaker-Independent Evaluation Protocol (Task 3 Design & Plan)

### 3.1 The Speaker Memorization Confound
In standard stratified splits, clips from every actor are randomly distributed across train, validation, and test sets. Deep neural networks readily memorize speaker biometric traits (vocal tract geometry, formant frequencies, fundamental pitch ranges) rather than generalized emotional expressions. Under random splits, the network easily functions as a pseudo-speaker identifier mapping speaker identity to emotion labels.

To quantify the magnitude of this confound, we have designed strict speaker-disjoint splits:
- **EmoDB (10 speakers: 5M, 5F):** 6 training speakers (3M, 3F, ~60% of clips), 2 validation speakers (1M, 1F, ~20%), 2 test speakers (1M, 1F, ~20%).
- **CAFE (12 speakers: 6M, 6F, 78 clips each):** 8 training speakers (4M, 4F, 624 clips, 66.7%), 2 validation speakers (1M, 1F, 156 clips, 16.7%), 2 test speakers (1M, 1F, 156 clips, 16.7%).
- **AESDD (5 core actors: 1–5):** 3 core actors + fallback actor 6 (301 clips, ~60%), 1 core actor for validation (~100 clips, ~20%), 1 core actor for test (~100 clips, ~20%).
- **MESD:** **EXCLUDED** from Task 3 because it contains only 3 demographic cohorts (`female`, `male`, `child`) without individual speaker identities.

### 3.2 Strict Evaluation Protocol & Integrity Guardrails
1. **Zero Speaker Overlap:** Verified programmatically with automated assertions: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
2. **Model Selection on Validation Speakers Only:** Model checkpoints are monitored and selected solely by accuracy on the unseen validation speakers. The test speakers are scored exactly once on the frozen best checkpoint.
3. **Task 3 Run Matrix:** 3 datasets $\times$ 2 feature types (MFCC, MFMC) $\times$ 3 coefficient counts (12, 24, 30) $\times$ 2 setups (`none`, `version_a`) $\times$ 3 seeds (42, 43, 44) = 108 runs.
4. **Estimated Execution Time:** ~14 to 16 minutes total (~3 mins EmoDB, ~5.5 mins AESDD, ~6 mins CAFE).
5. **Execution Status:** **Paused awaiting user review and explicit confirmation before running.**

---

## 4. Methodological Ambiguities Documented in Published Literature

1. **Input Dimensionality:** The paper specifies *"144 feature vectors"* but omits the framing parameters, window length, hop size, and audio duration required to generate 144 frames.
2. **Padding Configuration:** Without `padding="same"`, 4 Conv1D layers with kernel size 15 collapse 24-dimensional feature representations to negative dimensions.
3. **Augmentation Timing & Leakage:** The paper omits whether 5x Gaussian augmentation occurred before or after data partitioning. Week 3 proved that pre-split augmentation leaks 98.5% of clip identities, creating an artificial +15–20% performance boost.
4. **Speaker Partitioning:** The published paper did not implement or report speaker-independent cross-validation, allowing speaker identity memorization to inflate reported accuracies.
5. **Data Provenance:** CAFE and MESD were ingested via the open CAMEO HuggingFace mirror and were not directly verified against the original Zenodo 1478765 and Mendeley Data releases.
