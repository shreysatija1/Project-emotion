# Independent Reproduction: 1D-CNN Speech Emotion Recognition with MFCC/MFMC

**Target Research Article:**  
Thomas Mary Little Flower, Thirasama Jaya, and Sreedharan Christopher Ezhil Singh,  
*"Data augmentation using a 1D-CNN model with MFCC/MFMC features for speech emotion recognition"*,  
*Automatika: Journal for Control, Measurement, Electronics, Computing and Communications*, Vol. 65, No. 4, pp. 1297–1311 (2024).  
DOI: [10.1080/00051144.2024.2371249](https://doi.org/10.1080/00051144.2024.2371249)

---

## 1. Project Overview & Research Mandate

This repository contains the complete, rigorous, independent scientific reproduction of the 2024 *Automatika* paper on Speech Emotion Recognition (SER). The original paper claimed near-perfect classification accuracies (**97%–99%**) across multiple international corpora using a compact 1D-CNN and 5x additive Gaussian noise data augmentation.

Across five weeks and more than 1,000 controlled experimental runs across four speech databases (**EmoDB**, **AESDD**, **CAFE**, and **MESD**), our reproduction study identified the critical methodological factors behind the published claims:
1. **Pre-Split Data Leakage:** Augmenting audio before data partitioning leaks 95.9% of test clip source recordings into training, inflating accuracy by +15–20% artificially.
2. **144-Input Representation Flaw:** Preserving 144 temporal frames produces 576 units into `Dense(120)`, accelerating overfitting and dropping test accuracy to 51%–61%. The compact 24/48/60 baseline (mean + std) provides essential implicit regularization.
3. **Speaker Identity Confound:** Enforcing strict speaker disjointness drops test accuracy by **8.2 to 21.2 percentage points** (average collapse of **-16.5 percentage points**). The 1D-CNN functions primarily as a speaker biometric classifier under random splits.
4. **Regularization & Acoustic Augmentation Limits:** Neither L2 decay, SpatialDropout, nor pitch shifting/SpecAugment bridges the gap. Under leak-free, speaker-independent evaluation, true accuracy is **31% to 55%** (roughly 2x to 3.5x theoretical chance).

---

## 2. Environment Setup

The repository uses Python 3.11 with an isolated virtual environment (`.venv`).

### Activation

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### Dependencies
```bash
pip install -r requirements.txt
```

---

## 3. One-Command Reproduction

To generate and display all primary reproduction tables and verification summaries from the complete experimental logs:

```bash
python scripts/reproduce_all.py
```

This single command generates:
- `results/table1_reproduction_summary.csv`: Paper claims vs. leak-free stratified baseline vs. speaker-independent cross-validation across all datasets.
- `results/table4_regularization_study.csv`: Validation and test accuracies across regularized configurations on EmoDB.
- `results/table5_acoustic_augmentations.csv`: Cross-corpus speaker-independent evaluation of Pitch Shift, Time Stretch, and SpecAugment vs. Gaussian noise.
- `results/table6_statistical_significance.csv`: Paired statistical tests, 95% confidence intervals, and p-values.

---

## 4. Running Verification Tests

To verify data integrity, absence of NaNs, speaker disjointness, and augmentation leakage isolation:

```bash
python tests/test_week5.py
```

The test suite validates:
1. `test_feature_shapes_and_validity`: 144-input feature shapes and absence of NaNs/Infs.
2. `test_stratified_splits`: 60/20/20 split proportions and mutual exclusivity.
3. `test_speaker_independent_splits`: Zero speaker overlap across all rotating folds.
4. `test_acoustic_augmentations`: Pitch shifting, time stretching, and SpecAugment validity.
5. `test_augmentation_leakage_and_scaler_isolation`: Post-split training-only augmentation, zero test leakage, and train-only `StandardScaler` fitting.

---

## 5. Repository Structure

```
Project Emotion/
├── data/
│   ├── raw/
│   │   ├── emodb/           # Berlin Emotional Speech Database (535 clips, 7 emotions)
│   │   ├── cafe/            # Canadian French Emotional Speech (936 clips, 7 emotions)
│   │   ├── aesdd/           # Acted Emotional Speech Dynamic Database (500 clips, 5 emotions)
│   │   └── mesd/            # Mexican Spanish Emotional Database (862 clips, 6 emotions)
│   └── processed/           # Processed metadata and split files
├── features/                # Cached feature representations and augmentation caches
├── models/                  # Checkpoints for verified best-performing models
├── results/                 # All experimental result CSVs, plots, and tables
│   ├── week4_results.csv    # 618 verified Week 4 runs
│   ├── week5_results.csv    # Week 5 regularization & acoustic augmentation runs
│   ├── table1_reproduction_summary.csv
│   ├── table4_regularization_study.csv
│   ├── table5_acoustic_augmentations.csv
│   └── CHANGELOG_WEEK4_RECONCILIATION.md
├── scripts/                 # Execution, aggregation, and slide generation scripts
│   ├── reproduce_all.py     # Master one-command reproduction runner
│   ├── run_task1_regularization_parallel.py
│   ├── run_task2_augmentations_parallel.py
│   ├── statistical_comparison.py
│   ├── generate_week4_deck.py
│   └── generate_week5_deck.py
├── src/                     # Core library modules
│   ├── config.py            # Global paths and constants
│   ├── data_loader.py       # Multi-corpus audio ingestion and metadata
│   ├── extract_all_features.py # Feature extraction (MFCC/MFMC baseline & 144)
│   ├── features_144.py      # frames144 and stats144 implementations
│   ├── acoustic_augment.py  # Pitch shift, time stretch, and SpecAugment
│   ├── model.py             # 1D-CNN architecture specification
│   └── split_utils.py       # Stratified and speaker-independent split generators
├── tests/                   # Verification test suites
│   ├── test_week4.py
│   └── test_week5.py
├── docs/                    # Final reports, paper text, and specifications
│   ├── FINAL_REPRODUCTION_REPORT.md
│   └── spec_sheet.md
├── Week4_Status.pptx        # Week 4 Status Presentation (16:9 widescreen)
├── Week5_Final_Status.pptx  # Week 5 Final Status Presentation (16:9 widescreen)
├── requirements.txt         # Core dependencies
└── README.md                # Project documentation
```

---

## 6. Summary of Headline Findings

| Dataset | Language | Paper Claim (Table 3) | Our Stratified Baseline | Our Speaker-Independent | Generalization Drop |
|---|---|---|---|---|---|
| **EmoDB** | German | 86.9% / 97.2% | 71.0% ± 0.9% / 72.3% ± 3.3% | 51.6% ± 4.9% / 55.2% ± 6.8% | **-17.0 pts** |
| **AESDD** | Greek | 95.0% / 97.8% | 57.3% ± 7.2% / 64.3% ± 4.6% | 37.2% ± 3.9% / 44.0% ± 6.6% | **-18.9 pts** |
| **CAFE** | French | 93.6% / 99.5% | 50.0% ± 1.1% / 54.6% ± 6.7% | 34.3% ± 4.8% / 36.3% ± 5.2% | **-13.5 pts** |
| **MESD** | Spanish | 94.2% / 96.9% | 66.3% ± 4.3% / 69.0% ± 1.3% | *Excluded (Demographic cohorts)* | *N/A* |
