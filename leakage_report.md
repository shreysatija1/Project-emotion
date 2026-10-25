# Version B Data Leakage Analysis Report

## Executive Summary
Version B replicates the common methodological flaw where audio data augmentation is applied **prior** to partitioning into train, validation, and test splits (*augment-then-split*). This causes augmented variants of the exact same underlying audio recordings (and speakers) to appear across both the training set and the evaluation sets.

## Quantitative Findings

| Split | Total Rows | Rows with `source_clip_id` in Train | Row Leakage % | Unique Source Clips in Split | Sources Overlapping with Train | Source Clip Leakage % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 1,605 | 1,605 | 100.0% | 527 (out of 535 total) | 527 | 100.0% |
| **Validation** | 535 | **517** | **96.64%** | 365 | **357** | **97.81%** |
| **Test** | 535 | **513** | **95.89%** | 364 | **356** | **97.80%** |

## Key Insights
1. **Pervasive Cross-Contamination**: 
   - Over **96.6%** of validation samples and **95.8%** of test samples are augmented perturbations of recordings that the model was trained on.
   - 527 out of the 535 total EmoDB clips (98.50%) are present in the training partition in at least one augmented form.
2. **Explaining the Paper's 97.5% Accuracy Claim**:
   - The original Automatika (2024) paper achieved ~85% accuracy without data augmentation, but jumped to **97.5%** with Gaussian noise augmentation.
   - Because 95.9% of the test samples share the exact same speaker, pitch contours, sentence content, and recording acoustics as training samples, the model is largely performing memorization and clip-re-identification rather than generalizable emotion recognition.
3. **Contrast with Version A (split-then-augment)**:
   - In Version A, **0%** of validation and test clips appear in training (`Train & Val overlap: 0`, `Train & Test overlap: 0`).
