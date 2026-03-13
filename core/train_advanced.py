"""
Advanced Training Script for Breast Cancer Detection.

This script manages the Two-Stage training process for the Dual-Stream Ensemble model:
1.  Stage 1 (Head-Only): Trains only the classifier and attention blocks.
2.  Stage 2 (Fine-Tuning): Unfreezes backbones and trains the full model using
    Gradient Accumulation and the Muon optimizer for efficient convergence.

Key Features:
    - Deterministic Patient-Aware Split.
    - Focal Loss for Hard Negative Mining.
    - Muon Optimizer for Backbones.
    - Gradient Accumulation for effective batch size.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import copy
import time
from tqdm import tqdm

from .data_loader import BreastCancerDataset
from .advanced_model import AdvancedBreastCancerModel
from .advanced_stats import analyze_results
from .losses import FocalLoss
from .optimizers import Muon
from .metrics import calculate_metrics

def train_advanced_model(data_dir, num_epochs=30, batch_size=4):
    """
    Executes the training pipeline.
    
    Args:
        data_dir (str): Path to the dataset root.
        num_epochs (int): Total training epochs.
        batch_size (int): Physical batch size (virtual batch size = batch_size * accumulation_steps).
    """
    print(f"Initializing Advanced Dual-Stream Model on {data_dir}...")
    
    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Data Transformations (ImageNet Standard) ---
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # --- Dataset Initialization ---
    full_dataset = BreastCancerDataset(data_dir=data_dir, transform=transform)
    
    # --- Patient-Aware Split (Deterministic) ---
    # Ensures no patient appears in both Train and Val
    print("Performing Patient-Level Split...")
    samples = full_dataset.samples # (path, bbox, label, pid)
    pids = [s[3] for s in samples]
    unique_pids = np.unique(pids)
    
    # Deterministic Shuffle
    rng = np.random.RandomState(42)
    rng.shuffle(unique_pids)
    
    split_idx = int(len(unique_pids) * 0.8)
    train_pids = set(unique_pids[:split_idx])
    val_pids = set(unique_pids[split_idx:])
    
    # Save Split for reproducible evaluation
    np.save('val_patient_ids.npy', list(val_pids))
    print(f"Saved {len(val_pids)} validation patient IDs to 'val_patient_ids.npy'")
    
    train_indices = [i for i, s in enumerate(samples) if s[3] in train_pids]
    val_indices = [i for i, s in enumerate(samples) if s[3] in val_pids]
    
    print(f"Train Samples: {len(train_indices)} (Patients: {len(train_pids)})")
    print(f"Val Samples: {len(val_indices)} (Patients: {len(val_pids)})")
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=val_sampler, drop_last=False)
    
    # --- Model Initialization ---
    model = AdvancedBreastCancerModel().to(device)
    
    # --- Loss Function ---
    # Focal Loss minimizes the impact of easy negatives and focuses on hard examples.
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # --- Optimizer Setup (Initial) ---
    # Initially, we only train the head (Classifier + Attention)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-2)
    optimizers = optimizer 
    
    # Setting up ReduceLROnPlateau Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )
    
    best_auc = 0.0
    
    # --- Training Phases ---
    NUM_EPOCHS_HEAD = 5
    NUM_EPOCHS_FINE_TUNE = num_epochs - NUM_EPOCHS_HEAD
    
    print(f"Starting Training: {NUM_EPOCHS_HEAD} epochs Head-Only, then {NUM_EPOCHS_FINE_TUNE} Fine-Tuning.")
    
    best_model_wts = copy.deepcopy(model.state_dict()) 
    
    for epoch in range(num_epochs):
        # --- Phase Transition Check ---
        if epoch == NUM_EPOCHS_HEAD:
            print("\n>>> UNFREEZING BACKBONES for Fine-Tuning <<<")
            model.unfreeze_backbone()
            
            # Re-configure Optimizers for Fine-Tuning
            # Feature Extractors (Backbones) -> Muon Optimizer
            backbone_params = list(model.stream1.parameters()) + \
                              list(model.stream2.parameters()) + \
                              list(model.proj1.parameters()) + \
                              list(model.proj2.parameters())
            
            # Classifier Head -> AdamW
            head_params = list(model.fusion_attention.parameters()) + \
                          list(model.classifier.parameters())
            
            optimizer_backbone = Muon(backbone_params, lr=0.002) 
            optimizer_head = torch.optim.AdamW(head_params, lr=1e-4, weight_decay=1e-2)
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_head, mode='max', factor=0.5, patience=3, min_lr=1e-6
            )
            
            optimizers = [optimizer_backbone, optimizer_head]
            
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
                
            running_loss = 0.0
            running_corrects = 0
            
            all_labels = []
            all_preds = []
            
            train_pbar = tqdm(dataloader, desc=f"{phase.capitalize()} Epoch {epoch+1}")
            
            for i, (inputs, labels) in enumerate(train_pbar):
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                
                # Gradient Accumulation (Effective Batch Size Increase)
                accumulation_steps = 4 
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Normalize loss for accumulation
                    loss_step = loss / accumulation_steps 
                    
                    preds = torch.sigmoid(outputs)
                    
                    if phase == 'train':
                        loss_step.backward()
                        
                        is_last_batch = (i + 1) == len(dataloader)
                        if (i + 1) % accumulation_steps == 0 or is_last_batch:
                            if isinstance(optimizers, list):
                                for opt in optimizers:
                                    opt.step()
                                    opt.zero_grad()
                            else:
                                optimizers.step()
                                optimizers.zero_grad()
                        
                running_loss += loss.item() * inputs.size(0)
                train_pbar.set_postfix({'loss': loss.item()})
                
                # Binarize predictions for Accuracy tracking
                predicted_binary = (preds > 0.5).float()
                running_corrects += torch.sum(predicted_binary == labels.data)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.detach().cpu().numpy())
                
            epoch_loss = running_loss / len(dataloader.sampler)
            epoch_acc = running_corrects.double() / len(dataloader.sampler)
            
            print(f"  {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            if phase == 'val':
                val_metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
                print(f"  Val Metrics: {val_metrics}")
                
                # Save Best Model based on AUC
                if val_metrics['AUC'] > best_auc:
                    best_auc = val_metrics['AUC']
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), 'advanced_model_sota.pt')
                    print(f"  * New Best AUC: {best_auc:.4f} *")
                    
                
                # Plateau scheduler steps with validation metric (AUC)
                scheduler.step(val_metrics['AUC'])
                
    print(f"Training Complete. Best Val AUC: {best_auc:.4f}")
    
    # Reload best model for final check
    model.load_state_dict(best_model_wts)

if __name__ == '__main__':
    # Default data path
    DATA_PATH = r"C:\Users\dcaiafa\.gemini\antigravity\scratch\dataset"
    if os.path.exists(DATA_PATH):
        train_advanced_model(DATA_PATH)
    else:
        print(f"Dataset not found at {DATA_PATH}. Please Configure.")
