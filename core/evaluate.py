"""
Evaluation Script for Breast Cancer Detection.

This script performs the final validation of the trained model using:
1.  Patient-Aware Split (reloading `val_patient_ids.npy`).
2.  Test Time Augmentation (TTA) with Horizontal Flip.
3.  Threshold Optimization (F1-Score Maximization).
4.  Detailed Metrics Calculation (AUC, Sensitivity, Specificity).
"""

import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from breast_cancer_detection.data_loader import BreastCancerDataset, get_transforms
from breast_cancer_detection.advanced_model import AdvancedBreastCancerModel
from breast_cancer_detection.advanced_stats import analyze_results

def evaluate_model():
    """
    Loads the SOTA model and evaluates it on the validation set.
    Generates usage metrics and statistical analysis.
    """
    # --- Data Setup ---
    data_dir = r"C:\"
    model_path = r"C:\"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model from {model_path}...")
    model = AdvancedBreastCancerModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # --- Dataset Setup ---
    # We use 'val' transforms (No augmentation, just resize/normalize)
    full_dataset = BreastCancerDataset(data_dir=data_dir, transform=get_transforms('val'))
    
    # --- Reproducible Split Loading ---
    split_file = "val_patient_ids.npy"
    if os.path.exists(split_file):
        print(f"Loading consistent validation split from {split_file}...")
        val_pids = set(np.load(split_file))
    else:
        print("WARNING: Split file not found. Falling back to Seed 42.")
        rng = np.random.RandomState(42) 
        pids = [s[3] for s in full_dataset.samples]
        unique_pids = np.unique(pids)
        rng.shuffle(unique_pids)
        split_idx = int(len(unique_pids) * 0.8)
        val_pids = set(unique_pids[split_idx:])
    
    val_indices = [i for i, s in enumerate(full_dataset.samples) if s[3] in val_pids]
    print(f"Evaluating on {len(val_indices)} samples (Patients: {len(val_pids)})")
    
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    val_loader = DataLoader(full_dataset, batch_size=16, sampler=val_sampler)
    
    final_labels = []
    final_probs = []
    
    print("Running Inference (Enhanced 3-View TTA: Original + HFlip)...")
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            
            # --- Test Time Augmentation (TTA) ---
            
            # View 1: Original Image
            out1 = torch.sigmoid(model(inputs))
            
            # View 2: Horizontal Flip (medically valid)
            out2 = torch.sigmoid(model(torch.flip(inputs, [3])))
            
            # Ensemble Averaging
            probs = (out1 + out2) / 2.0
            
            final_labels.extend(labels.cpu().numpy())
            final_probs.extend(probs.cpu().numpy())
    
    y_true = np.array(final_labels).flatten()
    y_probs = np.array(final_probs).flatten()
    
    # --- Threshold Optimization ---
    # Find the operating point that maximizes F1-Score
    print("\n--- Threshold Optimization ---")
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_thresh = 0.5
    
    for thresh in thresholds:
        preds = (y_probs > thresh).astype(float)
        
        tp = np.sum((preds == 1) & (y_true == 1))
        fn = np.sum((preds == 0) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        tn = np.sum((preds == 0) & (y_true == 0))
        
        # Calculate Metrics (Adding Epsilon for numerical stability)
        sens = tp / (tp + fn + 1e-6)
        spec = tn / (tn + fp + 1e-6)
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
        print(f"Thresh={thresh:.2f} | Sens={sens:.3f} | Spec={spec:.3f} | F1={f1:.3f}")
        
    print(f"\n>> Best F1 Threshold: {best_thresh:.2f}")
    
    # --- Final Statistical Report ---
    analyze_results(y_true, y_probs, threshold=best_thresh)

if __name__ == '__main__':
    evaluate_model()
