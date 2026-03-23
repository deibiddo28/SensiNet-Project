"""
Visualization Module (Grad-CAM).

Provides Explainable AI (XAI) tools to visualize model focus.
Implements Gradient-Weighted Class Activation Mapping (Grad-CAM).
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple

class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping.
    
    Visualizes the regions of the input image that contributed most to the prediction
    by computing the gradient of the target class score with respect to the feature maps
    of the final convolutional layer.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Args:
            model (nn.Module): The PyTorch model instance.
            target_layer (nn.Module): The specific layer to hook (usually last Conv2d).
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register Hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """Forward hook to save feature maps."""
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        """Backward hook to save gradients."""
        self.gradients = grad_output[0] 

    def generate_cam(self, input_image: torch.Tensor, target_class: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """
        Generates Grad-CAM heatmap for a specific input image.
        
        Args:
            input_image (torch.Tensor): Preprocessed image [1, 3, H, W].
            target_class (int, optional): Target class index. Defaults to predicted class.
            
        Returns:
            Tuple[np.ndarray, float]: (Heatmap [H, W], Prediction Probability)
        """
        self.model.eval()
        
        # Enable gradients for the input (required for backward pass flow)
        input_image.requires_grad_(True)
        
        # Forward Pass
        output = self.model(input_image) # Logits [1, 1]
        
        # Define Target
        # For Binary Classification (Sigmoid):
        # We simply backward from the logit itself.
        target = output
        
        # Zero Gradients
        self.model.zero_grad()
        
        # Backward Pass
        target.backward(retain_graph=True)
        
        # Check if hooks fired
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not fire. Check if target_layer is in the forward path.")
            
        # Get Gradients and Activations
        gradients = self.gradients.detach().cpu().numpy()[0] # [C, H, W]
        activations = self.activations.detach().cpu().numpy()[0] # [C, H, W]
        
        # Global Average Pooling of Gradients (Importance Weights)
        weights = np.mean(gradients, axis=(1, 2)) # [C]
        
        # Weighted Combination of Feature Maps
        # Broadcasting: [C, 1, 1] * [C, H, W] -> Sum over C -> [H, W]
        cam = np.sum(weights[:, None, None] * activations, axis=0) 
        
        # ReLU: Keep only positive contributions
        cam = np.maximum(cam, 0)
        
        # Normalize and Resize
        cam = cv2.resize(cam, (input_image.shape[3], input_image.shape[2]))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8) # Avoid div zero
        
        return cam, torch.sigmoid(output).item()

def show_cam_on_image(img_tensor: torch.Tensor, cam_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlays coefficients heatmap on the original image.
    
    Args:
        img_tensor (torch.Tensor): Normalized image tensor [3, H, W].
        cam_mask (np.ndarray): Heatmap mask [H, W] in range [0, 1].
        alpha (float): Transparency factor.
        
    Returns:
        np.ndarray: Blended image [H, W, 3] in RGB.
    """
    # Denormalize Image (Approximate Mean/Std 0.5 for visualization)
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = img * 0.5 + 0.5 # [0, 1]
    img = np.clip(img, 0, 1)
    
    # Create Heatmap (Jet Colormap)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255.0
    
    # Overlay
    cam_result = heatmap * alpha + img * (1 - alpha)
    cam_result = np.clip(cam_result, 0, 1)
    
    return cam_result

if __name__ == '__main__':
    # --- Integration Test ---
    import os
    from breast_cancer_detection.advanced_model import AdvancedBreastCancerModel
    from breast_cancer_detection.data_loader import BreastCancerDataset, get_transforms
    
    print("Testing Grad-CAM Integration...")
    
    # 1. Load Model
    model = AdvancedBreastCancerModel()
    
    # 2. Identify Target Layer (Last Conv of Stream 1 - Xception)
    target_layer = None
    # Traverse to find the last Conv2d in stream1
    for name, module in model.stream1.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
            
    if target_layer is None:
        print("Error: Could not find a Conv2d layer in stream1.")
    else:
        print(f"Hooked into: {target_layer}")
        
        grad_cam = GradCAM(model, target_layer)
        
        # 3. Load Sample Image
        DATA_PATH = r"C:\"
        if os.path.exists(DATA_PATH):
            ds = BreastCancerDataset(DATA_PATH, subset='train', transform=get_transforms('val'))
            
            if len(ds) > 0:
                img, label = ds[0]
                input_tensor = img.unsqueeze(0) # [1, 3, 299, 299]
                
                try:
                    mask, prob = grad_cam.generate_cam(input_tensor)
                    print(f"Success! Prediction: {prob:.4f}, Label: {label}")
                except Exception as e:
                    print(f"Grad-CAM Failed: {e}")
        else:
            print("Dataset not found, skipping image test.")
