# Changelog: Week 4 Discrepancy Reconciliation & Headline Numbers Audit

**Target File:** `results/week4_results.csv` (618 experimental runs)  
**Audited Files:** `WEEK4_FINDINGS_NOTE.md`, `scripts/generate_week4_deck.py`, `Week4_Status.pptx`  
**Execution Date:** Week 5 Audit  

---

## 1. Speaker-Independent Drop Range Reconciliation

- **The Discrepancy:** The results table showed drops from -8.2 to -21.2 percentage points, but the note stated "12% to 21%", Slide 9 stated "15% to 25%", and Slide 10 stated "12% to 25%".
- **The Exact CSV Query:**
  ```python
  # Stratified baseline (Task 2, 24 & 30 coeffs): mean across 3 seeds per config
  t2 = df[(df["split_type"] == "stratified") & (df["format"] == "baseline") & (df["coeffs"].isin([24, 30]))]
  t2_mean = t2.groupby(["dataset", "feature", "coeffs", "setup"])["test_acc"].mean()

  # Speaker-independent (Task 3): average seeds within fold, then mean across folds
  t3 = df[df["split_type"] == "speaker_independent"]
  t3_fold = t3.groupby(["dataset", "feature", "coeffs", "setup", "fold"])["test_acc"].mean().reset_index()
  t3_mean = t3_fold.groupby(["dataset", "feature", "coeffs", "setup"])["test_acc"].mean()

  # Generalization Drop = spk_mean - strat_mean
  drop = (t3_mean - t2_mean) * 100
  ```
- **Audited Range:**
  * **Minimum Drop:** **-8.2 percentage points** (`cafe`, `mfcc`, 30 coeffs, `version_a`: 39.9% -> 31.7%).
  * **Maximum Drop:** **-21.2 percentage points** (`aesdd`, `mfcc`, 24 coeffs, `version_a`: 64.3% -> 43.1%).
  * **Mean Drop Across All 24 Configurations:** **-16.5 percentage points**.
  * **Per-Corpus Breakdown:**
    - `emodb`: -13.7 to -19.5 percentage points (mean: **-17.0 points**).
    - `aesdd`: -16.8 to -21.2 percentage points (mean: **-18.9 points**).
    - `cafe`: -8.2 to -18.4 percentage points (mean: **-13.5 points**).
- **Files Fixed:**
  * `WEEK4_FINDINGS_NOTE.md` Section 3.4: Replaced "12% to 21%" with "8.2 to 21.2 percentage point drop across all 24 evaluated configurations (average drop: -16.5 percentage points)".
  * `scripts/generate_week4_deck.py` Slide 9, Slide 10, and speaker notes: Harmonized to "8.2 to 21.2 percentage point drop across all configurations (average drop: -16.5 points)".

---

## 2. Augmentation Shift Range Reconciliation

- **The Discrepancy:** `WEEK4_FINDINGS_NOTE.md` stated "-1.5% to +3.5%", Slide 7 stated "-7% to +11%", and Slide 10 stated "-2% to +5%".
- **The Exact CSV Query:**
  ```python
  # Stratified baseline shift (Task 2, Version A minus None)
  t2_piv = t2_summary.pivot(index=["dataset", "feature", "coeffs"], columns="setup", values="strat_mean")
  t2_shift = (t2_piv["version_a"] - t2_piv["none"]) * 100

  # Speaker-independent shift (Task 3, Version A minus None)
  t3_piv = t3_summary.pivot(index=["dataset", "feature", "coeffs"], columns="setup", values="spk_mean")
  t3_shift = (t3_piv["version_a"] - t3_piv["none"]) * 100
  ```
- **Audited Numbers:**
  * **Stratified 24 & 30 Coeffs (Task 2 across 4 corpora):** Shift ranges from **-6.2 to +8.9 percentage points** (mean **+3.0 points**). (Including 12-coefficient EmoDB, MFMC-12 shifts +11.9 points from 51.7% to 63.6%, giving a range of **-6.2 to +11.9 points**).
  * **Speaker-Independent Folds (Task 3 across 3 corpora):** Shift ranges from **-2.6 to +9.7 percentage points** (mean **+2.7 points**).
  * **Combined Range Across All 24/30-Coeff Conditions:** **-6.2 to +9.7 percentage points**.
  * **Variance Context:** Typical standard deviations across seeds and folds range from **±2% to ±11%**, which completely encompasses these shifts.
- **Files Fixed:**
  * `WEEK4_FINDINGS_NOTE.md` Section 2.4: Clarified exact stratified shift (-6.2 to +8.9 pts) and speaker-independent shift (-2.6 to +9.7 pts).
  * `scripts/generate_week4_deck.py` Slide 7 and Slide 10: Standardized on "-6.2 to +8.9 percentage points (stratified) and -2.6 to +9.7 percentage points (speaker-independent)".

---

## 3. Data Leakage Percentage Harmonization

- **The Discrepancy:** Week 3 reported "95.9% of version-B test clips share a source recording with training", whereas Week 4 text cited "98.5%".
- **The Exact Query from `data/processed/version_b_metadata.csv`:**
  ```python
  tr_sources = set(meta_b[meta_b["split"] == "train"]["source_clip_id"])
  val_df = meta_b[meta_b["split"] == "val"]
  test_df = meta_b[meta_b["split"] == "test"]

  test_leak_pct = 100.0 * test_df["source_clip_id"].isin(tr_sources).sum() / len(test_df)
  total_source_in_tr_pct = 100.0 * len(tr_sources) / 535
  ```
- **Recomputed Truth:**
  * **Test Clips Leaked:** Exactly **513 out of 535 test clips (95.89% ~ 95.9%)** have their parent source recording present in the training partition.
  * **Source Clips in Training:** **527 out of 535 unique recordings (98.50% ~ 98.5%)** appear in the training split.
  * **Harmonized Metric:** Week 4 erroneously confused the percentage of source clips in train (98.5%) with the percentage of leaked test clips (95.9%).
- **Files Fixed:**
  * `WEEK4_FINDINGS_NOTE.md` Section 4: Corrected to "95.9% of test clip source recordings into the training set (with 98.5% of all source clips appearing in train)".
  * `scripts/generate_week4_deck.py` Slide 2 and speaker notes: Updated to "95.9% of test clips share source recordings with training".

---

## 4. Corpus-Specific Speaker-Independent Accuracy Scope

- **The Discrepancy:** Slide 10 claimed "~48-55% in speaker-independent conditions", but this range applied strictly to German EmoDB.
- **The Exact CSV Query:**
  ```python
  t3_fold.groupby("dataset")["test_acc"].agg(["min", "max"])
  ```
- **Audited Numbers:**
  * **EmoDB (5 folds):** 47.1% to 55.2% (~47–55%).
  * **AESDD (5 folds):** 36.2% to 45.9% (~36–46%).
  * **CAFE (6 folds):** 31.4% to 36.7% (~31–37%).
  * **Cross-Corpus Range:** **31.4% to 55.2%**.
- **Files Fixed:**
  * `scripts/generate_week4_deck.py` Slide 10: Explicitly scoped to "~47–55% on EmoDB and ~31–46% on AESDD and CAFE".

---

## 5. AESDD Benchmark Name Correction

- **The Discrepancy:** Slide 6 mistakenly labeled AESDD as "Att-HACK Greek".
- **Correction:** The canonical full name is **Acted Emotional Speech Dynamic Database (Greek)** (Vryzas et al., 2018). (Att-HACK is a distinct French-accented corpus).
- **Files Fixed:**
  * `scripts/generate_week4_deck.py` Slide 6: Updated to "AESDD (Acted Emotional Speech Dynamic Database, Greek)".

---

## 6. Overclaim Softening

- **The Issue:** Overclaims using words like "proves" or asserting that the paper relied on speaker overlap were scientifically overly definitive.
- **Adjustments Made:**
  * Changed "Proves the 1D-CNN heavily relies on memorizing speaker vocal tract geometry" -> "Strongly suggests the 1D-CNN relies substantially on speaker acoustic characteristics under random splits".
  * Changed "Speaker Identity Memorization is the Dominant Hidden Factor" -> "Speaker Identity Confound Explains a Substantial Portion of the Performance Gap".
  * Changed "The paper's claimed peak accuracies ... rely on both speaker overlap and pre-split data augmentation leakage" -> "...are consistent with the combined effects of speaker overlap (random splits) and potential pre-split data augmentation leakage".
  * Changed "Week 3 proved that..." -> "Week 3 demonstrated that...".
- **Files Fixed:**
  * `WEEK4_FINDINGS_NOTE.md` Sections 3.4 and 4.
  * `scripts/generate_week4_deck.py` Slides 9 and 10 and speaker notes.

---

## 7. Standard Deviation Comparability Footnote

- **The Addition:** A standardized footnote clarifying that cross-fold standard deviations and cross-seed standard deviations measure different error components:
  > *"Footnote on Standard Deviations: Speaker-independent standard deviations are computed across cross-validation folds ($N=5$ or $6$, capturing inter-speaker variance), whereas stratified standard deviations are computed across random seeds on a fixed split ($N=3$, capturing optimization/initialization variance); consequently, the two standard deviations are not directly comparable."*
- **Files Fixed:**
  * `WEEK4_FINDINGS_NOTE.md` Section 3.3.
  * `scripts/generate_week4_deck.py` Slides 7, 9, 10.
  * Regenerated `Week4_Status.pptx`.
