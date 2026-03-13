"""
Advanced Breast Cancer Detection Model Module.

This module defines a Dual-Stream Convolutional Neural Network (CNN) architecture
designed for high-performance breast cancer classification (Benign vs Malignant)
from mammography ROIs.

Architecture:
    - Stream 1: Xception (Legacy, Feature Rich)
    - Stream 2: EfficientNet-B3 (Modern, Efficient)
    - Fusion: Feature maps are projected, concatenated, and refined.
    - Attention: Convolutional Block Attention Module (CBAM) for feature refinement.

Classes:
    ChannelAttention: Squeeze-and-Excitation style channel attention.
    SpatialAttention: Spatial feature highlighting.
    CBAM: Combined Channel and Spatial Attention module.
    AdvancedBreastCancerModel: The main dual-stream ensemble.
"""

import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    Implements Channel Attention Module.
    
    Compresses spatial dimension to compute channel-wise importance weights.
    """
    def __init__(self, in_planes, ratio=16):
        """
        Args:
            in_planes (int): Number of input channels.
            ratio (int): Reduction ratio for the bottleneck.
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
    Implements Spatial Attention Module.
    
    Compresses channel dimension to compute spatial importance weights.
    """
    def __init__(self, kernel_size=7):
        """
        Args:
            kernel_size (int): Kernel size for the convolution.
        """
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Sequentially applies Channel Attention and Spatial Attention.
    """
    def __init__(self, planes):
        """
        Args:
            planes (int): Number of input channels.
        """
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class AdvancedBreastCancerModel(nn.Module):
    """
    Dual-Stream Ensemble Model for Breast Cancer Detection.
    
    Combines Xception and EfficientNet-B3 backbones, fuses their feature maps,
    applies CBAM attention, and classifies the result.
    """
    def __init__(self):
        super(AdvancedBreastCancerModel, self).__init__()
        
        # --- Stream 1: Xception (Legacy, Proven) ---
        # Output: 2048 channels
        self.stream1 = timm.create_model('xception', pretrained=True, num_classes=0)
        
        # --- Stream 2: EfficientNet-B3 (Modern, Efficient) ---
        # Output: 1536 channels
        self.stream2 = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        
        # --- Projection Layers ---
        # Reduce channel dimensions before fusion to balance contribution
        self.proj1 = nn.Conv2d(2048, 512, 1) # Xception -> 512
        self.proj2 = nn.Conv2d(1536, 512, 1) # EfficientNet -> 512
        
        # --- Attention Fusion ---
        # Input: 512 + 512 = 1024 channels
        self.fusion_attention = CBAM(1024)
        
        # --- Classification Head ---
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5), # Regularization
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input image batch [B, 3, 299, 299]
            
        Returns:
            logits (torch.Tensor): Raw output logits [B, 1]
        """
        # Feature Extraction
        f1 = self.stream1.forward_features(x) # [B, 2048, H1, W1]
        f2 = self.stream2.forward_features(x) # [B, 1536, H2, W2]
        
        # Spatial Alignment
        # Interpolate f2 to match f1 if dimensions differ (e.g. 10x10 vs 9x9)
        if f1.shape[2:] != f2.shape[2:]:
            f2 = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=False)
            
        # Projection
        p1 = self.proj1(f1) # [B, 512, H, W]
        p2 = self.proj2(f2) # [B, 512, H, W]
        
        # Fusion
        concat = torch.cat([p1, p2], dim=1) # [B, 1024, H, W]
        
        # Refinement
        refined = self.fusion_attention(concat)
        
        # Classification
        logits = self.classifier(refined)
        
        return logits

    def freeze_backbone(self):
        """Freezes gradients for both backbone streams (Transfer Learning)."""
        for param in self.stream1.parameters():
            param.requires_grad = False
        for param in self.stream2.parameters():
            param.requires_grad = False
        print("Backbones Frozen.")

    def unfreeze_backbone(self):
        """Unfreezes gradients for both backbone streams (Fine-Tuning)."""
        for param in self.stream1.parameters():
            param.requires_grad = True
        for param in self.stream2.parameters():
            param.requires_grad = True
        print("Backbones Unfrozen.")

if __name__ == '__main__':
    # Unit Test to verify shapes
    model = AdvancedBreastCancerModel()
    x = torch.randn(2, 3, 299, 299)
    out = model(x)
    print(f"Model Integrity Check Passed. Output Shape: {out.shape}")

