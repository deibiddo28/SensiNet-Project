"""
Loss Functions Module.

Contains custom loss implementations for Breast Cancer Detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection.
    
    References:
        Lin, T.-Y., et al. "Focal Loss for Dense Object Detection." ICCV 2017.
        
    Formula:
        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
        
    Attributes:
        alpha (float): Balancing factor for class imbalance (default: 1.0).
        gamma (float): Focusing parameter to down-weight easy examples (default: 2.0).
        reduction (str): 'mean', 'sum', or 'none'.
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs (torch.Tensor): Logits [B, C] (before sigmoid).
            targets (torch.Tensor): Ground truth labels [B] or [B, C] (0 or 1).
            
        Returns:
            torch.Tensor: Computed loss.
        """
        # Binary Cross Entropy with Logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # p_t: probability of the correct class
        # For BCE, pt = exp(-L) works because L = -log(pt)
        pt = torch.exp(-bce_loss) 
        
        # Focal Term
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

