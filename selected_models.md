# Model Selection Protocol (Frozen Before Test Evaluation)

**Strict Rule**: All models listed below were selected **strictly based on validation accuracy (`val_accuracy`)**.No test-set data was inspected or evaluated prior to finalizing this selection file.

| Configuration | Best Seed | Best Val Acc | Val Loss | Train Acc | Selected Model Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **MFCC** - 12 (after_split) | 42 | **72.90%** | 2.9602 | 96.39% | `best_model_mfcc_12_after_split_seed42.keras` |
| **MFCC** - 12 (before_split) | 42 | **83.18%** | 0.9067 | 98.57% | `best_model_mfcc_12_before_split_seed42.keras` |
| **MFCC** - 12 (none) | 42 | **67.29%** | 1.8452 | 100.00% | `best_model_mfcc_12_none_seed42.keras` |
| **MFCC** - 24 (after_split) | 42 | **72.90%** | 1.3147 | 96.51% | `best_model_mfcc_24_after_split_seed42.keras` |
| **MFCC** - 24 (before_split) | 43 | **89.72%** | 0.5699 | 99.88% | `best_model_mfcc_24_before_split_seed43.keras` |
| **MFCC** - 24 (none) | 43 | **72.90%** | 1.6361 | 100.00% | `best_model_mfcc_24_none_seed43.keras` |
| **MFCC** - 30 (after_split) | 42 | **73.83%** | 2.5352 | 99.94% | `best_model_mfcc_30_after_split_seed42.keras` |
| **MFCC** - 30 (before_split) | 43 | **91.40%** | 0.5786 | 99.88% | `best_model_mfcc_30_before_split_seed43.keras` |
| **MFCC** - 30 (none) | 43 | **72.90%** | 1.6210 | 100.00% | `best_model_mfcc_30_none_seed43.keras` |
| **MFMC** - 12 (after_split) | 44 | **73.83%** | 2.0563 | 95.64% | `best_model_mfmc_12_after_split_seed44.keras` |
| **MFMC** - 12 (before_split) | 43 | **77.57%** | 0.8880 | 96.57% | `best_model_mfmc_12_before_split_seed43.keras` |
| **MFMC** - 12 (none) | 43 | **62.62%** | 1.7941 | 99.07% | `best_model_mfmc_12_none_seed43.keras` |
| **MFMC** - 24 (after_split) | 43 | **77.57%** | 1.1866 | 95.39% | `best_model_mfmc_24_after_split_seed43.keras` |
| **MFMC** - 24 (before_split) | 42 | **85.61%** | 0.7038 | 99.69% | `best_model_mfmc_24_before_split_seed42.keras` |
| **MFMC** - 24 (none) | 42 | **71.96%** | 2.3334 | 98.75% | `best_model_mfmc_24_none_seed42.keras` |
| **MFMC** - 30 (after_split) | 43 | **74.77%** | 1.5905 | 96.82% | `best_model_mfmc_30_after_split_seed43.keras` |
| **MFMC** - 30 (before_split) | 43 | **88.04%** | 0.7039 | 100.00% | `best_model_mfmc_30_before_split_seed43.keras` |
| **MFMC** - 30 (none) | 43 | **71.96%** | 2.0861 | 100.00% | `best_model_mfmc_30_none_seed43.keras` |


## Per-Seed Validation Summary Table

| Feature | Coeffs | Aug Mode | Seed 42 | Seed 43 | Seed 44 | Mean Val Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| MFCC | 12 | `after_split` | 72.90% | 71.96% | 69.16% | **71.34%** |
| MFCC | 12 | `before_split` | 83.18% | 82.99% | 81.31% | **82.49%** |
| MFCC | 12 | `none` | 67.29% | 67.29% | 66.36% | **66.98%** |
| MFCC | 24 | `after_split` | 72.90% | 68.22% | 68.22% | **69.78%** |
| MFCC | 24 | `before_split` | 86.36% | 89.72% | 88.79% | **88.29%** |
| MFCC | 24 | `none` | 71.03% | 72.90% | 67.29% | **70.41%** |
| MFCC | 30 | `after_split` | 73.83% | 69.16% | 71.96% | **71.65%** |
| MFCC | 30 | `before_split` | 89.16% | 91.40% | 90.47% | **90.34%** |
| MFCC | 30 | `none` | 68.22% | 72.90% | 71.96% | **71.03%** |
| MFMC | 12 | `after_split` | 71.03% | 71.96% | 73.83% | **72.27%** |
| MFMC | 12 | `before_split` | 74.21% | 77.57% | 74.39% | **75.39%** |
| MFMC | 12 | `none` | 59.81% | 62.62% | 60.75% | **61.06%** |
| MFMC | 24 | `after_split` | 75.70% | 77.57% | 75.70% | **76.32%** |
| MFMC | 24 | `before_split` | 85.61% | 84.86% | 85.23% | **85.23%** |
| MFMC | 24 | `none` | 71.96% | 64.49% | 68.22% | **68.22%** |
| MFMC | 30 | `after_split` | 73.83% | 74.77% | 74.77% | **74.46%** |
| MFMC | 30 | `before_split` | 84.86% | 88.04% | 87.10% | **86.67%** |
| MFMC | 30 | `none` | 65.42% | 71.96% | 68.22% | **68.53%** |


*Status: FROZEN. Ready for single-pass test set scoring.*
