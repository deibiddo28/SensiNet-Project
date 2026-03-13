"""
Statistical Analysis Module.

Provides rigorous statistical tools for evaluating model performance,
including Bootstrap Confidence Intervals and Hypothesis Testing against benchmarks.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix
import scipy.stats as stats
from typing import Callable, Tuple

def bootstrap_metric(y_true: np.ndarray, y_pred: np.ndarray, metric_fn: Callable, n_bootstraps: int = 1000, ci: float = 0.95) -> Tuple[float, Tuple[float, float]]:
    """
    Computes a metric's confidence interval using non-parametric bootstrapping.
    
    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted values (probabilities or binary).
        metric_fn (Callable): Function (y_true, y_pred) -> float.
        n_bootstraps (int): Number of bootstrap samples.
        ci (float): Confidence interval width (e.g., 0.95).
        
    Returns:
        Tuple[float, Tuple[float, float]]: (Mean Score, (Lower CI, Upper CI))
    """
    bootstrapped_scores = []
    rng = np.random.RandomState(42)
    
    indices = np.arange(len(y_pred))
    
    for _ in range(n_bootstraps):
        # Sample with replacement
        pred_idx = rng.choice(indices, len(indices), replace=True)
        
        # Skip samples with only one class (metric undefined)
        if len(np.unique(y_true[pred_idx])) < 2:
            continue
            
        score = metric_fn(y_true[pred_idx], y_pred[pred_idx])
        bootstrapped_scores.append(score)
        
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    # Calculate Percentiles
    lower_bound = (1 - ci) / 2
    upper_bound = 1 - lower_bound
    
    ci_lower = sorted_scores[int(lower_bound * len(sorted_scores))]
    ci_upper = sorted_scores[int(upper_bound * len(sorted_scores))]
    
    return np.mean(bootstrapped_scores), (ci_lower, ci_upper)

def get_specificity(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Calculates Specificity (True Negative Rate)."""
    tn, fp, fn, tp = confusion_matrix(y_true, (y_pred >= threshold).astype(int), labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def get_sensitivity(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Calculates Sensitivity (Recall)."""
    return recall_score(y_true, (y_pred >= threshold).astype(int), zero_division=0)

def analyze_results(y_true: np.ndarray, y_probs: np.ndarray, specialist_sens: float = 0.87, threshold: float = 0.5):
    """
    Performs full statistical analysis and prints a report.
    
    Args:
        y_true (np.ndarray): Truth labels.
        y_probs (np.ndarray): Prediction probabilities.
        specialist_sens (float): Benchmark sensitivity to compare against.
        threshold (float): Decision threshold.
    """
    print(f"\n--- Statistical Analysis (Bootstrap 95% CI) @ Threshold {threshold} ---")
    
    # AUC
    auc_mean, auc_ci = bootstrap_metric(y_true, y_probs, lambda t, p: roc_auc_score(t, p))
    print(f"AUC:         {auc_mean:.4f} (95% CI: {auc_ci[0]:.4f} - {auc_ci[1]:.4f})")
    
    # Sensitivity
    sens_fn = lambda t, p: get_sensitivity(t, p, threshold)
    sens_mean, sens_ci = bootstrap_metric(y_true, y_probs, sens_fn)
    print(f"Sensitivity: {sens_mean:.4f} (95% CI: {sens_ci[0]:.4f} - {sens_ci[1]:.4f})")
    
    # Specificity
    spec_fn = lambda t, p: get_specificity(t, p, threshold)
    spec_mean, spec_ci = bootstrap_metric(y_true, y_probs, spec_fn)
    print(f"Specificity: {spec_mean:.4f} (95% CI: {spec_ci[0]:.4f} - {spec_ci[1]:.4f})")
    
    print("\n--- Comparison with Specialist Benchmark ---")
    print(f"Specialist Sensitivity Baseline: {specialist_sens}")
    
    # Simple Non-Parametric Test
    # If the CI lower bound > Benchmark, we are superior.
    # If the CI upper bound < Benchmark, we are inferior.
    if sens_ci[0] > specialist_sens:
        print(">> RESULT: Model is STATISTICALLY SUPERIOR to Specialist in Sensitivity.")
    elif sens_ci[1] < specialist_sens:
        print(">> RESULT: Model is Statistically Inferior to Specialist.")
    else:
        print(">> RESULT: No statistical difference (Competitive).")

if __name__ == '__main__':
    # Unit Test with dummy data
    y_true = np.random.randint(0, 2, 100)
    y_probs = np.random.rand(100)
    analyze_results(y_true, y_probs)

