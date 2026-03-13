"""
Metrics Module.

Provides utility functions to calculate standard classification metrics
incuding Sensitivity, Specificity, F1-Score, and AUC.
"""

import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
from typing import Dict

def calculate_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculates a suite of classification metrics.
    
    Args:
        y_true (np.ndarray): Ground truth binary labels (0 or 1).
        y_pred_prob (np.ndarray): Predicted probabilities [0, 1].
        threshold (float): Decision threshold for binarization.
        
    Returns:
        Dict[str, float]: Dictionary containing:
            - Sensitivity (Recall)
            - Specificity
            - Precision
            - F1
            - AUC
    """
    # Binarize Predictions
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    # Sensitivity (True Positive Rate)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    
    # Specificity (True Negative Rate)
    # confusion_matrix returns [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Precision & F1
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # AUC (Area Under Curve)
    try:
        # Requires at least one positive and one negative sample
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_pred_prob)
        else:
            auc = 0.5 # Default for single-class batches
    except ValueError:
        auc = 0.5
        
    return {
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'F1': f1,
        'AUC': auc
    }

