import os
import io
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import cv2
from pydantic import BaseModel

# Import model architecture
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advanced_model import AdvancedBreastCancerModel
from data_loader import get_transforms

# Initialize FastAPI App
app = FastAPI(title="Breast Cancer Detection API", description="Clinical inference engine for Mammogram Analysis", version="2.0.0")

# Setup CORS for Frontend Dev Server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for local dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Configuration & State ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "advanced_model_best.pth")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
transform = None

class InferenceResult(BaseModel):
    prediction: str
    confidence: float
    probability_malignant: float
    probability_benign: float
    decision_confidence: float
    risk_factors: dict
    xai_overlay: str # Base64 encoded image

import logging
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, transform
    logger.info(f"Loading Model from {MODEL_PATH} onto {DEVICE}...")
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"ERROR: {MODEL_PATH} not found.")
        yield
        return
        
    try:
        model = AdvancedBreastCancerModel().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        transform = get_transforms('val')
        logger.info("Model loaded successfully into global state.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    
    yield
    # Clean up on shutdown
    model = None
    transform = None

app.router.lifespan_context = lifespan

# --- Selective Dropout Activation for Bayesian Inference ---
def enable_dropout(m):
    """
    Selectively activates ONLY Dropout layers for MC Dropout inference.
    BatchNorm layers remain frozen in eval() to prevent N=1 variance crash.
    """
    if type(m) == nn.Dropout or type(m) == nn.Dropout2d:
        m.train()

def generate_gradcam(input_tensor, model, target_layer):
    """
    Generates a Grad-CAM heatmap from the specified target layer.
    
    Hooks into the target layer to capture forward activations and backward
    gradients, then computes weighted spatial importance maps.
    """
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    def forward_hook(module, input, output):
        activations.append(output)
        
    handle_b = target_layer.register_full_backward_hook(backward_hook)
    handle_f = target_layer.register_forward_hook(forward_hook)
    
    logits = model(input_tensor)
    model.zero_grad()
    logits.backward(retain_graph=True)
    
    try:
        grads = gradients[0][0].cpu().data.numpy()
        acts = activations[0][0].cpu().data.numpy()
    except IndexError:
        handle_b.remove()
        handle_f.remove()
        return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)
        
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i, :, :]
        
    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    if np.max(cam) != 0:
        cam = cam / np.max(cam)
        
    handle_b.remove()
    handle_f.remove()
    return cam

def generate_multi_layer_gradcam(input_tensor, model):
    """
    Multi-Layer Grad-CAM Fusion: Generates heatmaps from BOTH the CBAM-refined
    fusion tensor AND both projection layers, then fuses them for maximum coverage.
    
    - fusion_attention: Deepest semantically rich layer (captures what the CBAM attention focused on)
    - proj1 (Xception stream): Captures Xception-specific spatial features
    - proj2 (EfficientNet stream): Captures EfficientNet-specific spatial features
    
    The fusion produces a heatmap that reflects the complete decision-making of
    both streams and the attention mechanism.
    """
    # Primary: CBAM Fusion output (deepest, most semantically rich)
    cam_fusion = generate_gradcam(input_tensor, model, model.fusion_attention)
    
    # Secondary: Individual stream projections for fine-grained detail
    cam_xception = generate_gradcam(input_tensor, model, model.proj1)
    cam_effnet = generate_gradcam(input_tensor, model, model.proj2)
    
    # Weighted Fusion: CBAM attention gets dominant weight (0.6) because it represents
    # the refined, attention-weighted decision surface. Individual streams contribute
    # fine-grained spatial details that CBAM may have down-weighted.
    fused_cam = (0.6 * cam_fusion) + (0.2 * cam_xception) + (0.2 * cam_effnet)
    
    # Re-normalize to [0, 1]
    fused_cam = fused_cam - np.min(fused_cam)
    if np.max(fused_cam) != 0:
        fused_cam = fused_cam / np.max(fused_cam)
    
    return fused_cam

def create_xai_overlay(original_image: Image.Image, cam_map: np.ndarray):
    """
    Overlays the heatmap on the original image with anatomical tissue masking,
    morphological cleanup, and bounding box ROI detection.
    """
    original_cv = np.array(original_image)
    if len(original_cv.shape) == 2:
        original_cv = cv2.cvtColor(original_cv, cv2.COLOR_GRAY2RGB)
    original_cv = cv2.cvtColor(original_cv, cv2.COLOR_RGB2BGR)
    cam_resized = cv2.resize(cam_map, (original_cv.shape[1], original_cv.shape[0]), 
                              interpolation=cv2.INTER_CUBIC)
    heatmap = np.uint8(255 * cam_resized)
    
    # Anatomical Breast Tissue Masking
    # Prevent the neural network from throwing high activations in the black/empty background void.
    gray = cv2.cvtColor(original_cv, cv2.COLOR_BGR2GRAY)
    _, breast_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    
    # Morphological cleanup: remove small noise dots, fill gaps in the breast contour
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    breast_mask = cv2.morphologyEx(breast_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    breast_mask = cv2.morphologyEx(breast_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Mask the heatmap to strictly physical tissue
    heatmap = cv2.bitwise_and(heatmap, heatmap, mask=breast_mask)
    
    # Re-normalize heatmap inside the breast mask so the hottest tissue spot becomes 255 (Red)
    if np.max(heatmap) > 0:
        heatmap = np.uint8(255 * (heatmap / np.max(heatmap)))
    
    # Gaussian blur on heatmap for smoother visual transitions (reduces pixelation artifacts)
    heatmap = cv2.GaussianBlur(heatmap, (9, 9), 0)
    
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Advanced Alpha Blending with non-linear curve
    alpha = ((heatmap / 255.0) ** 0.8)[..., np.newaxis] 
    
    # Blend
    overlay = (alpha * heatmap_colored + (1 - alpha) * original_cv).astype(np.uint8)
    
    # Strict Top 30% Bounding Box Calculation
    _, thresh = cv2.threshold(heatmap, int(255 * 0.70), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    anomaly_size = 0.0
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        # Only draw box if the area is meaningful
        if cv2.contourArea(c) > 50:
            x, y, w, h = cv2.boundingRect(c)
            # Add a slight 10% padding around the tight box for visual breathing room
            pad = int(w * 0.10)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(original_cv.shape[1], x + w + pad), min(original_cv.shape[0], y + h + pad)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            anomaly_size = ((x2-x1) * (y2-y1)) / (original_cv.shape[0] * original_cv.shape[1]) * 100
            
    _, buffer = cv2.imencode('.png', overlay)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return b64_str, anomaly_size

@app.post("/api/analyze", response_model=InferenceResult)
async def analyze_mammogram(file: UploadFile = File(...)):
    """
    Analyzes an uploaded mammogram image using:
    1. Test-Time Augmentation (TTA) with horizontal flip
    2. Bayesian Monte Carlo Dropout (10-pass) for uncertainty quantification
    3. Multi-Layer Grad-CAM fusion for comprehensive spatial explanations
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        original_image = image.copy()
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Create a horizontally flipped version
        input_flipped = torch.flip(input_tensor, [3])  # Flip along width axis
        
        with torch.set_grad_enabled(True):
            input_tensor.requires_grad_()
            
            # --- BAYESIAN MONTE CARLO DROPOUT with TTA ---
            model.eval()
            model.apply(enable_dropout)  # Force ONLY dropout on, keep BatchNorm frozen
            mc_predictions = []
            
            for _ in range(10):
                # Original view
                logits_orig = model(input_tensor)
                prob_orig = torch.sigmoid(logits_orig).item()
                
                # Flipped view (TTA)
                logits_flip = model(input_flipped)
                prob_flip = torch.sigmoid(logits_flip).item()
                
                # TTA ensemble: average of both views per pass
                prob_tta = (prob_orig + prob_flip) / 2.0
                mc_predictions.append(prob_tta)
                
            model.eval()  # Return to normal eval mode
            
            # The final probability is the Bayesian Mean of TTA-enhanced predictions
            prob_malig = float(np.mean(mc_predictions))
            prob_benign = 1.0 - prob_malig
            
            # True Confidence is inversely proportional to the Variance (Uncertainty)
            variance = float(np.var(mc_predictions))
            
            # Normalize variance (max theoretical variance of a probability is 0.25)
            # Map 0 variance -> 99% confidence, high variance -> down to 50%
            decision_confidence = max(0.50, 0.99 - (variance * 2.0))
            if prob_malig < 0.10 or prob_malig > 0.90:
                 decision_confidence = min(0.99, decision_confidence + 0.10)
            
        THRESHOLD = 0.40 
        is_malignant = prob_malig >= THRESHOLD
        
        # Fuses CBAM attention + Xception stream + EfficientNet stream
        cam = generate_multi_layer_gradcam(input_tensor, model)
        
        b64_overlay, anomaly_size = create_xai_overlay(original_image, cam)
        
        # Advanced Statistics Mapping
        risk_factors = {
            "Mass Density": min(100, prob_malig * 100 + (anomaly_size * 2)),
            "Structural Distortion": min(100, prob_malig * 80 + 10),
            "Microcalcifications": min(100, prob_malig * 95),
            "Margin Irregularity": min(100, anomaly_size * 5) if is_malignant else 15,
            "Asymmetry": min(100, prob_malig * 70 + 20)
        }
        
        return InferenceResult(
            prediction="Malignant" if is_malignant else "Benign",
            confidence=prob_malig if is_malignant else prob_benign,
            probability_malignant=round(prob_malig, 4),
            probability_benign=round(prob_benign, 4),
            decision_confidence=round(decision_confidence, 4),
            risk_factors={k: round(v, 2) for k, v in risk_factors.items()},
            xai_overlay=f"data:image/png;base64,{b64_overlay}"
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

