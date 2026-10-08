# Paper Specification Sheet

**Paper**: *"Data augmentation using a 1D-CNN model with MFCC/MFMC features for speech emotion recognition"* (Automatika, 2024)

| Item | Paper's setting | My assumption | Status (Known / Assumed / Unclear) |
| :--- | :--- | :--- | :--- |
| **Split** | 60/20/20 | same | Known |
| **Augmentation** | Gaussian noise 0.01, 0.025, 0.05 plus a zero-noise copy (5x data) | not done yet (Week 2) | Known |
| **Features** | MFCC and MFMC with 12, 24, 30 coefficients | same | Known |
| **MFMC** | log10 of mel-weighted magnitude spectrum, no DCT | same | Known |
| **Model** | 4 Conv1D layers (8/16/32/62 filters, kernel 15, stride 1) each with BatchNorm, ReLU, MaxPool(2); dropout 0.3 after 4th conv; Dense 120; dropout 0.2; softmax output | not built yet (Week 2) | Known |
| **Training** | 100 epochs, learning rate 0.001 | Week 2 | Known |
| **Input vector size** | paper says 144 | mean+std over time = 2 x n_coeff (24, 48, 60) | Unclear |
| **4th conv filters** | paper says 62 | likely typo, assume 64 | Assumed |
| **Optimizer** | not stated | Adam | Assumed |
| **Batch size** | not stated | 32 | Assumed |
| **Augmentation before or after split** | not stated | test both in Week 2 (Protocol A augment-then-split, Protocol B split-then-augment-train-only) | Unclear |
| **Audio length** | not stated | no padding or trimming, pooling over true frames | Assumed |
| **Target accuracies on EmoDB** | MFCC no-aug ~85%, MFMC no-aug 86.9%, MFMC with aug 97.5% | same targets | Known |
