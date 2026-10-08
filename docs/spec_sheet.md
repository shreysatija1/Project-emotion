# Paper Specification Sheet

**Paper**: *"Data augmentation using a 1D-CNN model with MFCC/MFMC features for speech emotion recognition"* (Automatika, 2024)  
**Authors**: Thomas Mary Little Flower, Thirasama Jaya, Sreedharan Christopher Ezhil Singh  
**Target Dataset (Initial Reproduction)**: Emo-DB (Berlin Database of Emotional Speech, 7 emotions, 535 utterances)

---

## Specification Matrix

| Item | Paper's Setting | My Assumption / Proposed Default | Status |
| :--- | :--- | :--- | :--- |
| **Data Split** | 60% train / 20% validation / 20% test | 60% train, 20% val, 20% test (stratified by emotion / speaker consideration) | Known |
| **Data Augmentation** | Gaussian noise additions: Set 1 (Original), Set 2 (zero-noise permuted copy), Set 3 (Set 2 + 0.01 noise), Set 4 (Set 2 + 0.025 noise), Set 5 (Set 1 + 0.05 noise), yielding 5x total data | 5 copies per utterance: original, copy with 0 noise, and Gaussian white noise with standard deviations $\sigma \in \{0.01, 0.025, 0.05\}$ | Known |
| **Features** | MFCC and MFMC; 12, 24, and 30 coefficients tested | Implement extraction for 12, 24, and 30 coefficients for both MFCC and MFMC | Known |
| **MFMC Definition** | Log10 of mel-filterbank-weighted linear magnitude spectrum $\log_{10} \sum \|X(k)\| H_m(k)$ (not squared power spectrum), without DCT | Compute magnitude STFT spectrum $\|X(k)\|$, pass through triangular Mel filterbank, apply $\log_{10}$, no DCT compression | Known |
| **Model Architecture** | 4 Conv1D layers (filters: 8, 16, 32, 62; kernel: 15, stride: 1), each followed by BatchNorm, ReLU, and MaxPool1D(pool_size=2); Dropout(0.3) after 4th conv block; Dense(120, ReLU); Dropout(0.2); Dense(num_classes, Softmax) | 4 Conv1D blocks with BN, ReLU, MaxPool(2); Dropout(0.3); Dense(120); Dropout(0.2); Dense(7, softmax) | Known |
| **Training Epochs & LR** | 100 epochs, learning rate = 0.001 | 100 epochs, learning rate $1 \times 10^{-3}$ | Known |
| **Input Vector Size** | Paper states "One hundred forty-four feature vectors as input data for our CNN's first layer" | Propose e.g., summary statistics across frames (mean, std, min, max, deltas, delta-deltas) summing to 144, or fixed-length temporal framing (to be decided on Day 3) | Unclear |
| **4th Conv Filter Count** | Paper text and diagram state 62 filters ("sixty-two filters", "62, 1x5") | Assumed 64 (likely a typographical error for standard power-of-2 filter progression: 8, 16, 32, 64), test 64 and optionally 62 | Assumed |
| **Optimizer** | Not explicitly named in the paper text | Assumed Adam optimizer (default $\beta_1=0.9, \beta_2=0.999$, categorical cross-entropy loss) | Assumed |
| **Batch Size** | Not stated in the paper | Assumed 32 | Assumed |
| **Augmentation Split Protocol** | Not stated whether augmentation is applied before or after dataset splitting | Test both protocols: <br>• **Protocol A**: Augment entire corpus then split 60/20/20 (tests if paper results relied on data leakage) <br>• **Protocol B**: Split first into train/val/test, augment training set only (rigorous machine learning practice) | Unclear |
| **Target Accuracies (Emo-DB)** | • MFCC without augmentation: ~85% (85.0% for 30 coeffs, 86.0% for 24 coeffs) <br>• MFMC without augmentation: 86.9% <br>• MFMC with augmentation: 97.2% (Table 3) / 97.5% (Table 2) | Benchmark reproduced models against 85.0% (MFCC no DA), 86.9% (MFMC no DA), and 97.2–97.5% (MFMC with DA) | Known |
