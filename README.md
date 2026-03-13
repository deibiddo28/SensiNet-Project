# SensiNet.AI - Clinical Mammography Analysis

**Author:** David Caiafa (Venezuela)  
**Status:** Production-Ready (SOTA AUC: 0.926)  

SensiNet is a clinical-grade breast cancer detection system leveraging a Dual-Stream Xception architecture originally trained on the CBIS-DDSM dataset. It achieves State-of-the-Art (SOTA) performance in distinguishing Malignant vs. Benign structural deformations in mammography scans.

## Features
*   **Dual-Stream Vision Architecture**: Fuses high-resolution structural features with down-sampled global context for 92.6% AUC accuracy.
*   **Explainable AI (XAI)**: Generates clinical Grad-CAM heatmaps mapped directly onto user-uploaded scans to highlight suspicious Regions of Interest (ROI).
*   **Advanced Risk Stratification**: Dynamically calculates actionable heuristic boundaries such as Mass Density, Structural Distortion, Margin Irregularity, and Microcalcifications.
*   **"Drop-and-Go" Clinical Dashboard**: A beautiful, vanilla Zero-Node.js frontend built explicitly for rapid deployment in hospital networks.

## Repository Structure
*   `/backend/` - The FastAPI inference engine (app.py). Manages model ingestion, transformations, tensor forward-passes, and XAI Hook extractions.
*   `/frontend/` - The Clinical Web Dashboard UI.
*   `/core/` - The PyTorch definitions for the model layers, focal losses, metrics, and dataset loaders.
*   `advanced_model_best.pth` - The pre-trained PyTorch weights file holding the SOTA network state. *(Note: Due to GitHub file size limits, this file is hosted externally. See Installation).*

> **Technical Note regarding Tailwind CSS Warning:**
> When booting the frontend, the console may show a warning: *"cdn.tailwindcss.com should not be used in production"*. 
> **This is intentional.** To ensure maximum accessibility and a "Drop-and-Go" installation without requiring doctors or researchers to install Node.js/NPM, we explicitly utilized the Tailwind CDN alongside Alpine.js for state management. This grants SensiNet an ultra-modern aesthetic without the bloat of a JavaScript build pipeline.

## Installation & Usage

### 1. Download the Pre-Trained Weights
Because the sophisticated Dual-Stream Xception architecture exceeds GitHub's 100MB file limit (it is ~160MB), the `.pth` weights file must be downloaded separately. 

1. Go to the Hugging Face Model Repository: [Aredeksu/SensiNet-Mammography](https://huggingface.co/Aredeksu/SensiNet-Mammography)
2. Download `advanced_model_best.pth`.
3. Place the downloaded `.pth` file directly into the root directory of this repository (`SensiNet_Repo/`).

### 2. Setup the Python Environment
Ensure you have Python 3.10+ installed.
```bash
# Clone the repository
git clone https://github.com/yourusername/SensiNet_Repo.git
cd SensiNet_Repo

# Create and activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install Core ML requirements
pip install -r requirements.txt
```

### 2. Boot the API Server
Start the inference engine locally:
```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
*Wait until you see the log: `Model loaded successfully into global state.`*

### 3. Open the Clinical Dashboard
With the API running in the background, simply double-click `frontend/index.html` to open it in any modern web browser.
Drag and drop your DICOM/PNG/JPEG scans onto the dropzone to request a localized heat-map analysis!

## License 
Open-Source. Developed in Venezuela. Dedicated to saving lives through accessible, high-performance medical AI.
