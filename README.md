# Speech Emotion Recognition (SER) Paper Reproduction

Reproducing the research paper:  
**"Data augmentation using a 1D-CNN model with MFCC/MFMC features for speech emotion recognition"**  
*Automatika: Journal for Control, Measurement, Electronics, Computing and Communications* (2024).  
Authors: Thomas Mary Little Flower, Thirasama Jaya, and Sreedharan Christopher Ezhil Singh.  
DOI: [10.1080/00051144.2024.2371249](https://doi.org/10.1080/00051144.2024.2371249)

---

## Project Goal

The goal of this reproduction project is to reproduce and rigorously validate the findings of the 2024 Automatika paper on the **Emo-DB** (Berlin Database of Emotional Speech, 535 German utterances across 7 emotions). The study compares **MFCC** (Mel-Frequency Cepstral Coefficients) and **MFMC** (Mel-Frequency Magnitude Coefficients) with and without Gaussian noise data augmentation using a 1D Convolutional Neural Network.

---

## Environment Setup

The project uses Python 3.11 with an isolated virtual environment (`.venv`).

### Activate the Virtual Environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.\.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Reproduce the Week 1 Pipeline

To run the entire end-to-end data processing, feature extraction, sanity baselines, visualization, and validation pipeline in one step:

```bash
python src/run_all.py
```

This single command automatically executes:
1. `src/data.py`: Loads 535 `.wav` files at 16 kHz mono and generates `data/processed/metadata.csv`.
2. `src/make_splits.py`: Creates stratified 60/20/20 train/val/test splits (`data/processed/splits.csv`).
3. `src/features_mfcc.py`: Extracts 12, 24, and 30 coefficient MFCC features and saves them to `features/`.
4. `src/features_mfmc.py`: Extracts 12, 24, and 30 coefficient MFMC features and saves them to `features/`.
5. `src/sanity.py`: Trains SVM and Logistic Regression classifiers to verify signal quality.
6. `src/plots.py`: Plots class distribution and MFCC vs. MFMC spectrograms in `results/`.
7. `src/verify.py`: Runs assertion checks ensuring no split overlap, valid shapes, and feature consistency.

---

## Folder Structure

```
ser-repro/
├── data/
│   ├── raw/
│   │   └── emodb/           # 535 raw EmoDB .wav audio files
│   └── processed/           # metadata.csv and splits.csv
├── features/                # Cached extracted .npy feature arrays
├── src/                     # Source modules
│   ├── config.py            # Central settings, paths, and label mappings
│   ├── data.py              # Audio loader and metadata builder
│   ├── make_splits.py       # Stratified train/val/test split generator
│   ├── cache.py             # Feature and split loading/saving utilities
│   ├── features_mfcc.py     # MFCC extraction (12, 24, 30 coeffs)
│   ├── features_mfmc.py     # MFMC extraction (12, 24, 30 coeffs)
│   ├── sanity.py            # Baseline classifiers (SVM & LogReg)
│   ├── plots.py             # Plots class distribution & spectrograms
│   ├── verify.py            # Automated integrity and spot-check tests
│   └── run_all.py           # Master pipeline runner
├── results/                 # Sanity tables and output plots
├── docs/                    # Notes, spec sheet, and paper reference
│   ├── spec_sheet.md        # Paper parameter and assumption tracking
│   └── week1_notes.md       # Week 1 narrative and baseline findings
├── .gitignore               # Ignores .venv, datasets, features, and caches
├── requirements.txt         # Core dependencies
└── README.md
```

---

## Progress Log

### Week 1: Data Preparation, Feature Extraction & Sanity Baselines
- Configured Python 3.11 virtual environment with audio and scientific libraries (`numpy`, `scipy`, `pandas`, `librosa`, `soundfile`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`).
- Implemented `src/data.py` to parse 535 EmoDB files, extracting speaker, gender, emotion codes, and true audio durations.
- Generated stratified 60/20/20 splits (`train`: 321, `val`: 107, `test`: 107).
- Extracted and cached both MFCC and MFMC feature matrices across 12, 24, and 30 coefficients ($2 \times n$ pooled statistics).
- Ran baseline evaluations using SVM and Logistic Regression; test accuracies reached up to **88.79%** (MFCC-30) and **84.11%** (MFMC-30), decisively beating the majority-class baseline of **23.7%**.
- Generated exploratory visualizations and confirmed all data integrity assertions with `src/verify.py`.
- Formulated specification sheet (`docs/spec_sheet.md`) and Week 1 summary notes (`docs/week1_notes.md`).
