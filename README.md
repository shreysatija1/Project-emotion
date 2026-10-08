# Speech Emotion Recognition (SER) Paper Reproduction

Reproducing the paper:  
**"Data augmentation using a 1D-CNN model with MFCC/MFMC features for speech emotion recognition"**  
*Automatika: Journal for Control, Measurement, Electronics, Computing and Communications* (2024).  
Authors: Thomas Mary Little Flower, Thirasama Jaya, and Sreedharan Christopher Ezhil Singh.  
DOI: [10.1080/00051144.2024.2371249](https://doi.org/10.1080/00051144.2024.2371249)

---

## Project Goal

The primary goal of this 2-week effort is to reproduce the methodology and empirical findings of the paper, starting with the **Emo-DB** (Berlin Database of Emotional Speech) benchmark:
1. Re-implement feature extraction pipelines for:
   - **MFCC** (Mel-Frequency Cepstral Coefficients)
   - **MFMC** (Mel-Frequency Magnitude Coefficients) with magnitude spectrum log-filterbank weighting and no DCT.
2. Implement the Gaussian noise data augmentation pipeline (5x dataset size: original, zero-noise permuted, and $\sigma \in \{0.01, 0.025, 0.05\}$ noise levels).
3. Build and train the 4-layer 1D-CNN classifier under two split protocols (evaluating potential data leakage in augmentation vs. strict train-only augmentation).
4. Benchmark target accuracies on Emo-DB:
   - MFCC without augmentation: ~85.0%
   - MFMC without augmentation: 86.9%
   - MFMC with augmentation: 97.2% – 97.5%

---

## Environment Setup

The project uses Python 3.11 with virtual environment isolated in `.venv`.

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
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Smoke Test

To verify that all required audio, signal processing, and machine learning libraries are functional and that a basic Keras Conv1D training step executes cleanly:

```bash
python src/check_env.py
```

Expected output ends with:
```
ENVIRONMENT OK
```

---

## Folder Structure

```
ser-repro/
├── data/
│   ├── raw/             # Raw speech audio datasets (e.g., Emo-DB WAV files)
│   └── processed/       # Preprocessed audio waveforms or train/val/test splits
├── features/            # Cached extracted feature arrays (.npy)
├── src/                 # Python source code modules
│   └── check_env.py     # Environment verification and smoke test
├── notebooks/           # Jupyter exploratory notebooks
├── results/             # Evaluation metrics, confusion matrices, logs, and plots
├── docs/                # Documentation, paper PDF, and specification sheets
│   ├── paper.pdf        # Paper publication PDF
│   └── spec_sheet.md    # Known settings, assumptions, and unclear parameters
├── .gitignore           # Git ignore rules for virtual environment, data, and caches
├── README.md            # Project overview, setup, and progress log
└── requirements.txt     # Python project dependencies
```

---

## Progress Log

### Day 1: Project Setup & Environment Verification
- Initialized project directory structure with `.gitkeep` placeholders for tracked directories.
- Initialized Git repository and created comprehensive `.gitignore` ignoring `.venv`, `data/raw/`, `data/processed/`, `features/`, and cache artifacts.
- Created Python 3.11 virtual environment (`.venv`) and installed all core dependencies (`numpy`, `scipy`, `pandas`, `librosa`, `soundfile`, `scikit-learn`, `matplotlib`, `seaborn`, `tensorflow`, `jupyter`, `tqdm`).
- Created and executed `src/check_env.py` smoke test verifying synthetic 16 kHz audio generation, 30-band MFCC extraction via `librosa`, and a 1D-CNN training step via TensorFlow/Keras.
- Created `docs/spec_sheet.md` formalizing all known parameters, assumptions (Adam optimizer, batch size 32, 64-filter 4th conv layer), and unclear aspects (144-dimensional feature vector formulation, data augmentation split protocol A vs B).
