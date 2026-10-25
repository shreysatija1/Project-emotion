# Experimental Assumptions and Design Choices

This document tracks all explicit assumptions, paper ambiguities, and implementation decisions made throughout the reproduction of:
> *Thomas Mary Little Flower, Thirasama Jaya, and Sreedharan Christopher Ezhil Singh*,  
> **"Data augmentation using a 1D-CNN model with MFCC/MFMC features for speech emotion recognition"**,  
> *Automatika* (2024).

---

## 1. Canonical Feature File Naming
- **Choice**: Canonical naming for all augmented feature arrays, labels, and metadata is chosen as `version_a_*` and `version_b_*` (e.g., `features/version_a_mfcc_12.npy`, `features/version_a_labels.npy`, `data/processed/version_a_metadata.csv`).
- **Backward Compatibility**: The duplicate files prefixed with `emodb_version_*` are retained on disk to preserve any legacy references, but all training, evaluation, and downstream code exclusively reference the canonical `version_a_*` / `version_b_*` paths.

---

## 2. Audio Processing and Noise Scaling
- **Sample Rate**: All clips loaded at 16,000 Hz mono via Librosa, without artificial zero-padding or length trimming.
- **Additive Noise**: 
  - Augmented sets are created with additive white Gaussian noise directly scaled on waveform amplitude:
    $$\text{wave}_{\text{aug}} = \text{wave}_{\text{clean}} + \sigma \cdot \mathcal{N}(0, 1)$$
  - Noise levels ($\sigma$):
    - `set1`: $0.0$ (original waveform)
    - `set2`: $0.0$ (exact zero-noise copy of original)
    - `set3`: $0.01$
    - `set4`: $0.025$
    - `set5`: $0.05$
  - Fixed seed `SEED = 42` using `np.random.default_rng(42)`.

---

## 3. Feature Extraction & Input Lengths
- **MFCC and MFMC Pooling**:
  - STFT parameters: `n_fft = 512`, `hop_length = 256` (16 ms hop, 32 ms window at 16 kHz).
  - Mel bands: exactly $n_{\text{coeff}}$ mel filterbank bands are computed for both MFCC and MFMC ($12, 24, 30$).
  - Temporal pooling: mean and standard deviation across time frames concatenated $\to$ vector length of $2 \times n_{\text{coeff}}$:
    - 12 coefficients $\to$ 24 features
    - 24 coefficients $\to$ 48 features
    - 30 coefficients $\to$ 60 features
- **Paper Discrepancy on Input Dimension (144 features)**:
  - The paper claims the input layer has 144 features, yet states experiments evaluated 12, 24, and 30 coefficients. Mean and std over time naturally produce 24, 48, and 60.
  - Feeding 24, 48, and 60 directly into the 1D-CNN allows isolating the direct acoustic representation without inventing arbitrary extra statistics or framing that the authors did not specify.

---

## 4. Model Architecture & Conv1D Layer Shapes
- **Padding ('same')**:
  - The paper specifies 4 Conv1D layers with kernel size 15 and stride 1, followed each time by MaxPooling1D with pool size 2.
  - If `valid` padding were used, an input length of 24 would reduce to $24 - 15 + 1 = 10 \xrightarrow{\text{pool}} 5 < 15$, causing a negative dimension error at the second convolutional layer.
  - Therefore, `padding='same'` is mathematically required and used across all 4 Conv1D layers.
- **Layer 4 Filter Count (64 vs 62)**:
  - The paper text states "sixty-two" (62) filters for Conv1D block 4, following 8, 16, and 32 filters.
  - Because 8, 16, 32 are successive powers of 2, 62 is almost certainly a typographical error for 64.
  - Our implementation defaults to 64 filters (with support for 62 via an optional argument).
- **Flatten Layer Size Difference**:
  - With 4 successive $\text{MaxPool1D}(2)$ layers and `same` padding:
    - Input 24 $\to 12 \to 6 \to 3 \to 1$. Final feature map shape: $(1, 64) \xrightarrow{\text{flatten}} \mathbf{64}$ units.
    - Input 48 $\to 24 \to 12 \to 6 \to 3$. Final feature map shape: $(3, 64) \xrightarrow{\text{flatten}} \mathbf{192}$ units.
    - Input 60 $\to 30 \to 15 \to 7 \to 3$. Final feature map shape: $(3, 64) \xrightarrow{\text{flatten}} \mathbf{192}$ units.
  - **Important Note**: The 12-coefficient model feeds 64 inputs to the penultimate Dense(120) layer, whereas the 24- and 30-coefficient models feed 192 inputs. This structural difference in the classification head parameter count (49,687 vs 65,047 total params) must be taken into account when comparing performance across coefficient sizes.

---

## 5. Training Protocol
- **Optimizer & Hyperparameters**:
  - Optimizer: Adam with learning rate $0.001$.
  - Batch size: 32.
  - Max epochs: 100.
  - Loss function: Sparse Categorical Cross-Entropy.
  - Model checkpointing: Best model weights selected strictly by maximum validation accuracy (`val_accuracy`).
- **Data Standardization**:
  - Standardized via `StandardScaler` fitted **only on the training split**. Validation and test sets are transformed using the training mean and standard deviation to strictly prevent information leakage.
- **Test Set Scoring**:
  - Evaluated on test data strictly after model selection is finalized and frozen in `selected_models.md`.
