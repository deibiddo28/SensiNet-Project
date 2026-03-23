"""
Data Loading Module for Breast Cancer Detection.

Handles the loading, parsing, and preprocessing of the CBIS-DDSM dataset.
Supports:
1.  ROI extraction from generic image files using Supervisely JSON annotations.
2.  Context-aware cropping (padding around ROIs).
3.  On-the-fly transformations (Augmentation).
4.  Patient ID extraction for leak-free splitting.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import zlib
import base64
import cv2
import io

# --- Configuration ---
IMG_HEIGHT = 299
IMG_WIDTH = 299
BATCH_SIZE = 16 

# ImageNet Normalization Constants
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

class BreastCancerDataset(Dataset):
    """
    PyTorch Dataset for CBIS-DDSM Breast Cancer ROIs.
    
    Parses proprietary JSON annotations to extract Bounding Boxes (Bitmaps)
    and Class Labels (Malignant/Benign).
    """
    def __init__(self, data_dir, subset='train', transform=None, context_padding=0.2):
        """
        Args:
            data_dir (str): Root directory of the dataset.
            subset (str): Subdirectory to scan ('train' or 'test').
            transform (callable, optional): Transform to be applied on a sample.
            context_padding (float): Percentage of context to include around the ROI.
        """
        self.data_dir = data_dir
        self.subset = subset
        self.transform = transform
        self.context_padding = context_padding
        self.samples = [] # List of (img_path, bbox, label, patient_id)
        
        self.ann_dir = os.path.join(data_dir, subset, 'ann')
        self.img_dir = os.path.join(data_dir, subset, 'img')
        
        self._scan_dataset()

    def _decode_bitmap_bbox(self, bitmap_obj):
        """Decodes Supervisely ZLIB-Base64 bitmap to find Bounding Box (x1, y1, x2, y2)."""
        try:
            encoded = bitmap_obj['data']
            origin = bitmap_obj['origin']
            
            # Decompress
            compressed = base64.b64decode(encoded)
            decompressed = zlib.decompress(compressed)
            
            # Convert to Mask
            n = np.frombuffer(decompressed, np.uint8)
            mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
            
            if mask is None:
                return None
                
            # Handle Alpha Channel
            if len(mask.shape) > 2:
                mask = mask[:, :, -1]
                
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            
            if not np.any(rows) or not np.any(cols):
                return None
                
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Calculate Absolute Coords
            x1 = cmin + origin[0]
            y1 = rmin + origin[1]
            x2 = cmax + origin[0]
            y2 = rmax + origin[1]
            
            return (x1, y1, x2, y2)
        except Exception:
            return None

    def _scan_dataset(self):
        """Scans the annotation directory and populates self.samples."""
        if not os.path.exists(self.ann_dir):
            print(f"Annotation directory not found: {self.ann_dir}")
            return

        print(f"Scanning High-Res ROIs in {self.ann_dir}...")
        json_files = [f for f in os.listdir(self.ann_dir) if f.endswith('.json')]
        
        for json_file in json_files:
            json_path = os.path.join(self.ann_dir, json_file)
            image_name = json_file.replace('.json', '')
            image_path = os.path.join(self.img_dir, image_name)
            
            if not os.path.exists(image_path):
                continue
                
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                # Parsing Logic: Find 'Malignant' or 'Benign' tag
                img_label = -1
                for tag in data.get('tags', []):
                    tname = tag.get('name', '').lower()
                    if 'malignant' in tname:
                        img_label = 1.0
                        break
                    if 'benign' in tname:
                        img_label = 0.0
                        break
                    if tname == 'patient_id':
                         patient_id = tag.get('value', 'unknown')
                
                if img_label == -1:
                    continue # Skip unlabelled
                
                # Extract ROIs from objects
                for obj in data.get('objects', []):
                    if 'bitmap' in obj:
                        bbox = self._decode_bitmap_bbox(obj['bitmap'])
                        if bbox:
                            # Use found patient_id or fallback to filename prefix
                            pid = patient_id if 'patient_id' in locals() else image_name.split('_')[2]
                            self.samples.append((image_path, bbox, img_label, pid))
                            
            except Exception as e:
                print(f"Error parse {json_file}: {e}")
                
        print(f"Found {len(self.samples)} ROI samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Loads and transforms a sample.
        
        Returns:
            roi (Tensor): Transformed image tensor.
            label (Tensor): Float tensor (0.0 or 1.0).
        """
        img_path, bbox, label, _ = self.samples[idx]
        
        try:
            # Lazy Loading with PIL
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                w, h = img.size
                
                x1, y1, x2, y2 = bbox
                
                # Apply Context Padding
                roi_w = x2 - x1
                roi_h = y2 - y1
                pad_x = int(roi_w * self.context_padding)
                pad_y = int(roi_h * self.context_padding)
                
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)
                
                roi = img.crop((x1, y1, x2, y2))
                
                if self.transform:
                    roi = self.transform(roi)
                    
                return roi, torch.tensor(label, dtype=torch.float32)
                
        except Exception as e:
            # Robust Fallback for Corrupt Images
            print(f"Error loading ROI {idx}: {e}")
            return torch.zeros((3, IMG_HEIGHT, IMG_WIDTH)), torch.tensor(label, dtype=torch.float32)

def get_transforms(phase='train'):
    """Factory for Training/Validation transforms."""
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])

def get_dataloaders(data_dir, batch_size=BATCH_SIZE):
    """Utility to get a simple train loader."""
    train_dataset = BreastCancerDataset(data_dir, subset='train', transform=get_transforms('train'))
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

if __name__ == '__main__':
    # Unit Test - Put the dataset here 
    DATA_PATH = r"C:\" 
    if os.path.exists(DATA_PATH):
        ds = BreastCancerDataset(DATA_PATH)
        if len(ds) > 0:
            img, lbl = ds[0]
            print(f"Data Loader Integrity Check Passed. ROI Shape: {img.shape}, Label: {lbl}")
    else:
        print("Data Path not found, skipping specific tests.")

