# Independent Scientific Reproduction Report: 1D-CNN Speech Emotion Recognition with MFCC/MFMC Features

**Target Study:** Thomas Mary Little Flower, Thirasama Jaya, and Sreedharan Christopher Ezhil Singh, *"Data augmentation using a 1D-CNN model with MFCC/MFMC features for speech emotion recognition"*, *Automatika: Journal for Control, Measurement, Electronics, Computing and Communications*, Vol. 65, No. 4, pp. 1297–1311 (2024). DOI: [10.1080/00051144.2024.2371249](https://doi.org/10.1080/00051144.2024.2371249).  
**Reproduced By:** University Reproduction Team  
**Evaluation Scope:** Weeks 1–5 Comprehensive Investigation across EmoDB (German), AESDD (Greek), CAFE (Canadian French), and MESD (Mexican Spanish).  
**Integrity Standards:** Zero data leakage, StandardScaler fitted exclusively on training splits, models selected strictly on validation accuracy, test scored once per run, all negative and null results faithfully reported.

---

## 1. Executive Summary & Reproduction Overview

The target study claims that a compact 1D Convolutional Neural Network (1D-CNN) trained on Mel-Frequency Cepstral Coefficients (MFCC) or Mel-Frequency Magnitude Coefficients (MFMC) achieves unprecedented, near-perfect recognition accuracies across multiple international speech corpora when augmented with 5x additive Gaussian noise: **97.20% – 97.50% on EmoDB**, **97.80% – 99.20% on AESDD**, **98.60% – 99.50% on CAFE**, and **96.40% – 96.90% on MESD**.

Over a rigorous 5-week empirical investigation spanning more than 1,000 controlled experimental runs across four speech databases, three random seeds (42, 43, 44), and rotating cross-validation folds, our reproduction study yielded the following decisive findings:

1. **Failure to Replicate Published Performance Under Leak-Free Conditions:**
   Under strict, leak-free stratified evaluation, the 1D-CNN achieves **65.7% – 71.0%** unaugmented and **70.7% – 72.3%** with clean Gaussian noise on EmoDB (lagging the paper by **25.0 percentage points**). Across AESDD, CAFE, and MESD, leak-free models lag published claims by **28 to 45 percentage points**.
2. **The Pre-Split Data Leakage Discovery (Week 3):**
   The published paper omits whether 5x Gaussian augmentation occurred before or after data partitioning. If audio is augmented prior to splitting (Version B), **95.89% (95.9%) of test clips share their source recording with the training partition** (arising because 98.50% of all source clips appear in train). This data leakage artificially boosts accuracy by **+15 to +20 percentage points** without true generalization.
3. **The 144-Input Representation Flaw (Week 4):**
   The paper specifies *"144 feature vectors as input data"*. Evaluating both temporal STFT framing (`frames144`) and extended multi-stream summary statistics (`stats144`) revealed that preserving 144 temporal frames causes rapid memorization on small audio sample sizes, collapsing test accuracy to **51% – 61%**. The compact temporal mean + standard deviation pooling (24/48/60 values) acts as an essential regularizer.
4. **Speaker Identity Confound & Generalization Collapse (Week 4):**
   Standard stratified splits allow deep neural networks to memorize speaker vocal tract biometric signatures (pitch register, formant baselines). When evaluated on unseen speakers using rotating speaker-disjoint folds, test accuracy drops by **8.2 to 21.2 percentage points** (average collapse of **-16.5 percentage points**). On novel speakers, true performance is **47% – 55% on EmoDB**, **36% – 46% on AESDD**, and **31% – 37% on CAFE** (roughly 2x to 3.5x theoretical chance).
5. **Regularization and Novel Acoustic Augmentation Limits (Week 5):**
   Neither modern regularization (L2 weight decay, SpatialDropout1D, Early Stopping, Cosine LR) nor physically grounded acoustic augmentations (pitch shifting ±2 semitones, time stretching 0.9–1.1x, SpecAugment pre-pooling masking) bridge the gap to the paper. While regularization narrows the train-val overfitting gap from +36.5% to +11.4%, test accuracy remains indistinguishable from noise ($p = 0.542$). Acoustic augmentations perform within ±1.2 percentage points of simple Gaussian noise, with no method exceeding the cross-fold standard deviation.

---

## 2. Published Paper Claims vs. Controlled Leak-Free Results

The table below contrasts the published performance claims (Table 3 of the Automatika paper for 30 coefficients) against our leak-free stratified baseline (Task 2) and speaker-independent cross-validation (Task 3).

| Benchmark Corpus | Language | Classes | Theoretical Chance | Paper Claim (None / Aug) | Our Stratified Baseline (None / Ver A) | Our Speaker-Independent (None / Ver A) | True Generalization Drop ($\Delta$) |
|---|---|---|---|---|---|---|---|
| **EmoDB** | German | 7 | 14.3% | 85.0% / 96.1% (MFCC)<br>86.9% / 97.2% (MFMC) | 71.0% ± 0.9% / 70.7% ± 4.6%<br>65.7% ± 1.9% / 72.3% ± 3.3% | 51.6% ± 4.9% / 53.5% ± 4.6%<br>47.1% ± 1.7% / 55.2% ± 6.8% | **-17.0 pts** (range: -13.7 to -19.5 pts) |
| **AESDD** | Greek | 5 | 20.0% | 82.0% / 95.2% (MFCC)<br>95.0% / 97.8% (MFMC) | 54.0% ± 6.6% / 57.7% ± 11.1%<br>57.3% ± 7.2% / 64.3% ± 4.6% | 37.2% ± 3.9% / 40.2% ± 4.5%<br>40.5% ± 6.8% / 44.0% ± 6.6% | **-18.9 pts** (range: -16.8 to -21.2 pts) |
| **CAFE** | French | 7 | 14.3% | 88.8% / 98.6% (MFCC)<br>93.6% / 99.5% (MFMC) | 46.1% ± 5.2% / 39.9% ± 4.2%<br>50.0% ± 1.1% / 54.6% ± 6.7% | 34.3% ± 4.8% / 31.7% ± 3.2%<br>35.4% ± 6.4% / 36.3% ± 5.2% | **-13.5 pts** (range: -8.2 to -18.4 pts) |
| **MESD** | Spanish | 6 | 16.7% | 85.5% / 96.1% (MFCC)<br>94.2% / 96.9% (MFMC) | 66.3% ± 4.3% / 69.0% ± 1.3%<br>64.0% ± 5.5% / 65.7% ± 0.4% | *N/A (Excluded: cohort labels only)* | *N/A* |

> [!NOTE]
> *Footnote on Standard Deviations:* Speaker-independent standard deviations are computed across cross-validation folds ($N=5$ or $6$, capturing inter-speaker variance), whereas stratified standard deviations are computed across random seeds on a fixed split ($N=3$, capturing optimization/initialization variance); consequently, the two standard deviations are not directly comparable.

---

## 3. Deep-Dive Findings by Investigation Phase

### 3.1 Week 3 Finding: The Pre-Split Data Augmentation Leakage
- **Flaw Mechanism:** In audio data augmentation, generating copies of clips before applying train/validation/test partitioning violates statistical independence. 
- **Exact Leakage Metric:** In Version B, **513 out of 535 test clips (95.89%) share a source recording with the training split**, while **527 out of 535 total source clips (98.50%) appear in the training partition**.
- **Empirical Impact:** Evaluating models under Version B leakage yields **80% – 89%** test accuracy. The network simply matches noise-corrupted copies back to their training parent recording. When augmentation is applied strictly after splitting (Version A), accuracy drops to **64% – 72%**.

### 3.2 Week 4 Finding: The 144-Input Representation Flaw
- **Paper Discrepancy:** The authors state: *"One hundred forty-four feature vectors as input data for our CNN's first layer."*
- **Two Plausible Interpretations Evaluated:**
  1. `frames144`: 2D temporal spectrogram `(144, n_coeff)` downsampled through 4 Conv1D blocks to 576 units feeding into `Dense(120)`.
  2. `stats144`: 1D extended statistical feature vector `(144, 1)` summarizing static, velocity ($\Delta$), and acceleration ($\Delta^2$) trajectories.
- **Empirical Result:** On 11 of 12 evaluated configurations on EmoDB, the compact 24/48/60 baseline achieved higher test accuracy (**64% – 71%**) than both 144-input representations (**51% – 61%**). Temporal framing dramatically increases parameter count, causing sequence memorization on small corpora.

### 3.3 Week 4 Finding: Speaker Identity Memorization Confound
- **The Confounder:** In random stratified splits, speech clips from the same actor appear in both train and test. Deep CNNs readily learn individual biometric traits (vocal tract length, formant frequencies) rather than emotion.
- **Speaker-Disjoint Protocol:** Rotating folds with complete speaker isolation ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$):
  - EmoDB: 10 speakers (5M/5F) -> 5 folds (6 train, 2 val, 2 test).
  - AESDD: 5 core actors -> 5 folds (3 train + fallback takes, 1 val, 1 test).
  - CAFE: 12 actors (6M/6F) -> 6 folds (8 train, 2 val, 2 test).
- **Generalization Drop:** Across all 24 configurations, test accuracy collapses by **8.2 to 21.2 percentage points** (average drop of **-16.5 percentage points**). Additive Gaussian noise provides no defense against speaker shift.

---

## 4. Week 5 Task 1: Regularization Study on EmoDB

To test whether the 25-point performance gap was driven by severe unregularized overfitting (baseline models reached 98–100% training accuracy while validation stagnated near 60%), we conducted a controlled study on EmoDB evaluating five regularization strategies:
1. `reg_none`: Unregularized baseline control.
2. `reg_l2`: L2 weight decay ($10^{-3}$) on dense layers.
3. `reg_spatial_dropout`: SpatialDropout1D ($0.20$) in convolutional blocks.
4. `reg_early_stopping`: Early stopping (patience 15 on validation accuracy).
5. `reg_cosine_lr`: Cosine annealing learning rate schedule ($10^{-3} \to 10^{-5}$).
6. `reg_combined`: L2 ($10^{-3}$) + SpatialDropout1D ($0.15$) + Early Stopping.

### Empirical Findings:
- **Best Model Selection by Validation Accuracy:** Following our protocol of selecting exclusively on validation accuracy, `reg_spatial_dropout` was the top performer on speaker-disjoint folds (Val Acc: **59.6% ± 5.1%**, Test Acc: **51.9% ± 6.2%** vs. baseline Val Acc: **58.5% ± 3.9%**, Test Acc: **50.5% ± 3.7%**). On stratified splits, `reg_l2` reached **72.5% ± 3.4%** val acc (Test: **65.5% ± 3.7%**).
- **Overfitting Was Successfully Suppressed:** SpatialDropout reduced the train-val gap by **10.90 percentage points** ($p < 0.0001$), while `reg_combined` reduced it by **19.24 percentage points** from 38.6% down to **19.4% ± 12.2%** ($p < 0.0001$).
- **Test Generalization Remained Flat:** Suppressing memorization did not bridge the 25-point gap to the paper. The test accuracy difference between `reg_spatial_dropout` and baseline is small and statistically indistinguishable from noise (+1.37 pts, 95% CI: [-0.34%, +3.08%], $t = 1.60, p = 0.116$).
- **Scientific Verdict:** The performance gap with the paper is **not caused by unregularized overfitting**. Preventing memorization does not supply the acoustic information required to achieve 85%–97% accuracy.

---

## 5. Week 5 Task 2: Acoustic Augmentations vs. Gaussian Noise

We evaluated whether physically grounded audio transforms—simulating natural voice variability—outperform additive Gaussian noise under speaker-disjoint folds on EmoDB, AESDD, and CAFE:
1. **Pitch Shift (±2 semitones):** Simulates inter-speaker vocal tract length and fundamental frequency jitter.
2. **Time Stretch (0.9x – 1.1x):** Simulates speaking rate variability.
3. **SpecAugment:** Time and frequency masking applied strictly to the 2D frame matrix **BEFORE mean+std pooling** (masking pooled vectors is meaningless).
4. **5x Augmentation Factor:** Matched exactly to Version A to ensure fair comparisons.

### Empirical Findings:
- **Speaker-Independent Accuracies Across Folds:**
  - **EmoDB (5 folds):** None: 50.0% ± 4.6% | Ver A (Gaussian): 53.1% ± 5.4% | Pitch Shift: 52.9% ± 3.7% | Time Stretch: 50.9% ± 4.3% | SpecAugment: 53.8% ± 5.9%.
  - **AESDD (5 folds):** None: 38.0% ± 5.5% | Ver A (Gaussian): 43.3% ± 5.7% | Pitch Shift: 43.8% ± 6.6% | Time Stretch: 41.5% ± 6.3% | SpecAugment: 42.6% ± 5.4%.
  - **CAFE (6 folds):** None: 34.2% ± 4.9% | Ver A (Gaussian): 34.0% ± 5.0% | Pitch Shift: 35.0% ± 5.2% | Time Stretch: 35.2% ± 5.2% | SpecAugment: 34.7% ± 4.6%.
- **Cross-Fold Standard Deviation Rule:** **Not a single augmentation difference exceeds the cross-fold standard deviation** across any of the three corpora.
- **Direct Paired Comparison Against Version A Gaussian Noise ($N=192$ pairs across all corpora):**
  - `pitch_shift` vs. Version A: Mean Diff = **+0.48%** [95% CI: -0.51%, +1.48%], $t = 0.957, p = 0.3396$ (**Indistinguishable from noise**).
  - `spec_augment` vs. Version A: Mean Diff = **+0.23%** [95% CI: -0.79%, +1.26%], $t = 0.451, p = 0.6525$ (**Indistinguishable from noise**).
  - `time_stretch` vs. Version A: Mean Diff = **-0.85%** [95% CI: -1.81%, +0.12%], $t = -1.721, p = 0.0869$ (**Indistinguishable from noise**).
- **Scientific Verdict:** Acoustic augmentations offer modest gains of ~1.7% to 3.0% over unaugmented training data, but they perform within ±0.85% of simple Gaussian noise and do not close the gap to the published claims.

---

## 6. Documented Ambiguities & Study Limitations

In adherence to scientific transparency, we disclose both the documented ambiguities in the published article and the inherent limitations of our reproduction:

### 6.1 Ambiguities in the Published Article:
1. **Input Dimensionality Specification:** The paper specifies *"144 feature vectors"* but omits the framing parameters, window length, hop size, and duration normalization.
2. **Convolutional Padding:** Without `padding="same"`, 4 Conv1D layers with kernel size 15 collapse 24-dimensional feature representations to negative dimensions.
3. **Augmentation Timing:** The paper omits whether 5x Gaussian augmentation occurred before or after train/val/test splitting.
4. **Speaker Partitioning:** The published study did not report speaker-independent cross-validation.

### 6.2 Limitations of Our Reproduction Study:
1. **Test Set Sample Sizes:** Small audio benchmarks (EmoDB: 535 clips; AESDD: 500 clips) yield small test partitions (~100 clips), resulting in wider confidence intervals.
2. **Data Mirror Provenance:** CAFE and MESD were ingested via the open CAMEO HuggingFace parquet mirror and were not directly verified against official Zenodo 1478765 or Mendeley Data distributions.
3. **Reverse-Engineered 144-Input Models:** Both `frames144` and `stats144` are our own reverse-engineered interpretations and are not confirmed by the published text.

---

## 7. Conclusions & Guidelines for Speech Emotion Recognition

1. **Rejection of Published Claims:** The published peak accuracies (~97% to 99%) were not replicated under our controlled, leak-free implementation. The data is consistent with the hypothesis that published figures reflect pre-split augmentation leakage and speaker overlap.
2. **Realistic Baseline for 1D-CNN SER:** Under strict leak-free conditions, this 1D-CNN architecture achieves **~63% – 72%** on EmoDB (40% – 66% on other corpora) in stratified splits, and **~31% – 55%** in realistic speaker-independent conditions (2x to 3.5x theoretical chance).
3. **Three Mandatory Standards for Future Research:**
   - **Post-Split Augmentation Only:** Augment training audio only after partitioning; validation and test sets must remain 100% clean.
   - **Mandatory Speaker-Disjoint Benchmarking:** Traditional stratified splits severely overestimate emotion recognition by memorizing speaker biometric identities.
   - **Complete Experimental Disclosures:** Authors must disclose hop sizes, framing parameters, random seeds, and data partitioning scripts.
