import os
import cv2
import json
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION
# ==========================================
class CONFIG:
    # Paths (Update these to match your exact directory structure)
    test_images_path = "/kaggle/input/recodai-luc-scientific-image-forgery-detection/test_images"
    sample_sub_path = "/kaggle/input/recodai-luc-scientific-image-forgery-detection/sample_submission.csv"
    
    # Model Weights Paths
    model1_path = "/kaggle/input/modelsbest309base/best_model.pth"
    model2_path = "/kaggle/input/dinobestmodel/pytorch/default/1/dino197.pth"
    dino_path = "/kaggle/input/dinov2/pytorch/base/1" # Local path to DINOv2 files
    
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 512
    use_tta = True         # Test Time Augmentation
    min_area_percent = 0.05 
    min_confidence = 0.336 

# ==========================================
# 2. MODEL ARCHITECTURES
# ==========================================

# --- Model 1: Custom CNN (Encoder-Decoder) ---
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 1, 1),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=(CONFIG.img_size, CONFIG.img_size), mode='bilinear', align_corners=False)
        return x

# --- Model 2: DINOv2 Segmenter ---
class Decoder(nn.Module):
    def __init__(self, in_ch=768, out_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, out_ch, 1)
        )
    
    def forward(self, f, size):
        return self.net(F.interpolate(f, size=size, mode="bilinear", align_corners=False))

class DinoSegmenter(nn.Module):
    def __init__(self, encoder, processor):
        super().__init__()
        self.encoder = encoder
        self.processor = processor
        self.seg_head = Decoder(768, 1)
    
    def forward_features(self, x):
        # Convert tensor back to numpy format expected by HuggingFace processor
        imgs = (x * 255).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        inputs = self.processor(images=list(imgs), return_tensors="pt").to(x.device)
        
        with torch.no_grad():
            feats = self.encoder(**inputs).last_hidden_state
        
        B, N, C = feats.shape
        # Exclude CLS token and reshape
        fmap = feats[:, 1:, :].permute(0, 2, 1)
        s = int(math.sqrt(N-1))
        fmap = fmap.reshape(B, C, s, s)
        
        return fmap
    
    def forward_seg(self, x):
        fmap = self.forward_features(x)
        return self.seg_head(fmap, (CONFIG.img_size, CONFIG.img_size))

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def load_model(model_path):
    """Safely loads models (supports both custom CNN and DINO based on path/dict)."""
    print(f"Loading model from: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=CONFIG.device)
        
        # Handle state dictionaries vs full models
        if isinstance(checkpoint, dict):
            state_dict = None
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                # If the checkpoint contains the full model object
                model = checkpoint['model']
                if hasattr(model, 'eval'):
                    model.eval()
                return model.to(CONFIG.device)
            else:
                state_dict = checkpoint
            
            # Try loading as DINO first
            try:
                processor = AutoImageProcessor.from_pretrained(CONFIG.dino_path, local_files_only=True)
                encoder = AutoModel.from_pretrained(CONFIG.dino_path, local_files_only=True).eval().to(CONFIG.device)
                model = DinoSegmenter(encoder, processor).to(CONFIG.device)
                if state_dict is not None:
                    model.load_state_dict(state_dict, strict=False)
                model.eval()
                return model
            except:
                # Fallback to Custom CNN
                try:
                    model = Model().to(CONFIG.device)
                    if state_dict is not None:
                        model.load_state_dict(state_dict, strict=False)
                    model.eval()
                    return model
                except Exception as e:
                    print(f"Failed to load architecture for {model_path}: {e}")
                    return None
        
        # If checkpoint is already a model object
        elif hasattr(checkpoint, 'eval'):
            checkpoint.eval()
            return checkpoint.to(CONFIG.device)
        
        return None
    except Exception as e:
        print(f"Error loading {Path(model_path).name}: {e}")
        return None

def predict_with_tta(model, image_tensor):
    """Predicts with Test Time Augmentation (Horizontal Flip, Vertical Flip, Rotation)."""
    predictions = []
    
    # 1. Original
    with torch.no_grad():
        if hasattr(model, 'forward_seg'):
            pred = torch.sigmoid(model.forward_seg(image_tensor))
        else:
            pred = torch.sigmoid(model(image_tensor))
    predictions.append(pred)
    
    if CONFIG.use_tta:
        # 2. Horizontal Flip
        with torch.no_grad():
            if hasattr(model, 'forward_seg'):
                pred = torch.sigmoid(model.forward_seg(torch.flip(image_tensor, dims=[3])))
            else:
                pred = torch.sigmoid(model(torch.flip(image_tensor, dims=[3])))
        predictions.append(torch.flip(pred, dims=[3]))
        
        # 3. Vertical Flip
        with torch.no_grad():
            if hasattr(model, 'forward_seg'):
                pred = torch.sigmoid(model.forward_seg(torch.flip(image_tensor, dims=[2])))
            else:
                pred = torch.sigmoid(model(torch.flip(image_tensor, dims=[2])))
        predictions.append(torch.flip(pred, dims=[2]))
        
        # 4. Rotation 90 degrees
        with torch.no_grad():
            if hasattr(model, 'forward_seg'):
                pred = torch.sigmoid(model.forward_seg(torch.rot90(image_tensor, 1, [2, 3])))
            else:
                pred = torch.sigmoid(model(torch.rot90(image_tensor, 1, [2, 3])))
        predictions.append(torch.rot90(pred, -1, [2, 3]))
        
        return torch.stack(predictions).mean(0)[0, 0].detach().cpu().numpy()
    
    return predictions[0][0, 0].detach().cpu().numpy()

def postprocess(pred, original_size):
    """Applies Gaussian Blur, Dynamic Thresholding, and Morphological operations."""
    # Smooth prediction
    pred = cv2.GaussianBlur(pred, (3, 3), 0)
    
    # Dynamic thresholding based on stats
    mean_val = np.mean(pred)
    std_val = np.std(pred)
    thr = mean_val + 0.3 * std_val
    mask = (pred > thr).astype(np.uint8)
    
    # Clean up small noise and fill holes
    if mask.sum() > 0:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 30: # Remove small blobs
                mask[labels == i] = 0
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    
    # Resize back to original image dimensions
    mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
    return mask

def rle_encode(mask):
    """Run Length Encoding for submission."""
    pixels = mask.T.flatten()
    dots = np.where(pixels == 1)[0]
    
    if len(dots) == 0:
        return "authentic"
    
    run_lengths = []
    prev = -2
    
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    
    return json.dumps([int(x) for x in run_lengths])

# ==========================================
# 4. MAIN INFERENCE LOOP
# ==========================================

def main():
    print(f"Device: {CONFIG.device}")
    
    # 1. Load Models
    models = {}
    model1 = load_model(CONFIG.model1_path)
    model2 = load_model(CONFIG.model2_path)
    
    if model1:
        models['model1'] = model1
        print("Model 1 loaded successfully.")
    if model2:
        models['model2'] = model2
        print("Model 2 loaded successfully.")
        
    if not models:
        print("CRITICAL WARNING: No models loaded. Check paths.")
        
    # 2. Prepare for Inference
    predictions = []
    
    # Get test files
    if os.path.exists(CONFIG.test_images_path):
        image_files = sorted([f for f in os.listdir(CONFIG.test_images_path) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))])
        print(f"Found {len(image_files)} test images.")
    else:
        print("Test directory not found. Using empty list.")
        image_files = []

    # 3. Iterate over images
    print("Starting inference...")
    for image_name in image_files:
        image_path = Path(CONFIG.test_images_path) / image_name
        
        # Load and Preprocess Image
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size
        total_pixels = original_width * original_height
        
        # Dynamic threshold for minimum area
        min_pixels_threshold = int(total_pixels * CONFIG.min_area_percent / 100.0)
        
        # Resize for model input
        image_array = np.array(image.resize((CONFIG.img_size, CONFIG.img_size)), np.float32) / 255
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)[None].to(CONFIG.device)
        
        # Ensemble Prediction
        ensemble_preds = []
        for model in models.values():
            pred = predict_with_tta(model, image_tensor)
            ensemble_preds.append(pred)
        
        # Average the predictions from both models
        if ensemble_preds:
            final_pred = np.mean(ensemble_preds, axis=0)
        else:
            final_pred = np.zeros((CONFIG.img_size, CONFIG.img_size))
            
        # Post-process
        mask = postprocess(final_pred, (original_width, original_height))
        mask_pixels = int(mask.sum())
        
        # Calculate confidence inside the mask
        if mask_pixels > 0:
            mask_resized = cv2.resize(mask, (CONFIG.img_size, CONFIG.img_size), interpolation=cv2.INTER_NEAREST)
            # Check mean confidence only within the predicted positive area
            mean_inside = float(final_pred[mask_resized == 1].mean()) if (mask_resized == 1).any() else 0.0
        else:
            mean_inside = 0.0
            
        # Decision Logic: Authentic vs Forged
        if mask_pixels < min_pixels_threshold or mean_inside < CONFIG.min_confidence:
            annotation = "authentic"
        else:
            annotation = rle_encode(mask)
            
        predictions.append({
            "case_id": Path(image_name).stem,
            "annotation": annotation
        })
        
    # 4. Generate Submission File
    print("Generating submission.csv...")
    predictions_df = pd.DataFrame(predictions)
    predictions_df["case_id"] = predictions_df["case_id"].astype(str)
    
    if os.path.exists(CONFIG.sample_sub_path):
        submission = pd.read_csv(CONFIG.sample_sub_path)
        submission["case_id"] = submission["case_id"].astype(str)
        
        # Merge predictions with sample submission to ensure correct order
        submission = submission[["case_id"]].merge(predictions_df[["case_id", "annotation"]], 
                                                 on="case_id", 
                                                 how="left")
        
        submission["annotation"] = submission["annotation"].fillna("authentic")
        submission[["case_id", "annotation"]].to_csv("submission.csv", index=False)
        print("Submission saved successfully.")
    else:
        print("Sample submission file not found. Saving predictions directly.")
        predictions_df.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()