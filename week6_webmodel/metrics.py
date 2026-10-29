"""
Metrics module for Week 6:
- Macro-F1 (unweighted average F1 across classes)
- UAR (Unweighted Average Recall / balanced accuracy)
- Accuracy
- Expected Calibration Error (ECE)
"""

import numpy as np
from sklearn.metrics import f1_score, recall_score, accuracy_score, confusion_matrix


def compute_ece(y_true, y_probs, n_bins=10):
    """
    Compute Expected Calibration Error (ECE) for multi-class classification.
    
    Parameters
    ----------
    y_true : np.ndarray (N,)
        Integer true class labels.
    y_probs : np.ndarray (N, C)
        Predicted class probability distributions.
    n_bins : int
        Number of confidence bins (default 10).
        
    Returns
    -------
    float
        ECE metric (0.0 to 1.0).
    """
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return float(ece)


def evaluate_predictions(y_true, y_probs):
    """
    Compute complete primary and secondary evaluation metrics.
    
    Returns
    -------
    dict:
        'macro_f1', 'uar', 'accuracy', 'ece', 'conf_matrix'
    """
    y_pred = np.argmax(y_probs, axis=1)
    
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    uar = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    ece = compute_ece(y_true, y_probs, n_bins=10)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(y_probs.shape[1])))
    
    return {
        "macro_f1": macro_f1,
        "uar": uar,
        "accuracy": acc,
        "ece": ece,
        "conf_matrix": cm.tolist()
    }


if __name__ == "__main__":
    y_t = np.array([0, 1, 2, 0, 1, 2])
    y_p = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.1, 0.8],
        [0.3, 0.4, 0.3],
        [0.2, 0.6, 0.2],
        [0.1, 0.3, 0.6]
    ])
    res = evaluate_predictions(y_t, y_p)
    print("Metrics test:", res)
