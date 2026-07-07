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
from transformers import AutoModel
import warnings
import gc

# Suppress warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION
# ==========================================
class CONFIG:
    # --- Paths ---
    test_images_path = "/kaggle/input/recodai-luc-scientific-image-forgery-detection/test_images"
    sample_sub_path = "/kaggle/input/recodai-luc-scientific-image-forgery-detection/sample_submission.csv"
    
    # --- Model Weights ---
    # UPDATE THIS PATH to your uploaded .pth file
    model_path = "/kaggle/input/dinoexteme/other/default/1/dinov2_large_a40_extreme.pth"
    
    # --- Offline Config Path ---
    # Point this to the offline DINOv2-Large files on Kaggle.
    # If you haven't added the "dinov2" dataset, add it from the sidebar.
    # Path usually looks like: /kaggle/input/dinov2/pytorch/large/1
    dino_config_path = "/kaggle/input/dinov2/pytorch/large/1" 
    
    # --- Inference Settings ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 1024        # Must match training size (1024x1024)
    tile_size = 1024       # Sliding window size
    tile_overlap = 0.25    # Overlap to smooth edges
    batch_size = 4         # Keep small for T4 GPU inference
    min_confidence = 0.50  # Confidence threshold
    min_area = 0.05        # Minimum mask area to count as forgery

# ==========================================
# 2. MODEL ARCHITECTURE (DINOv2 Large)
# ==========================================
class DinoLargeSegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        # Load Offline to prevent ConnectionError
        print(f"Loading DINOv2 backbone from: {CONFIG.dino_config_path}")
        try:
            self.encoder = AutoModel.from_pretrained(CONFIG.dino_config_path, local_files_only=True)
        except Exception as e:
            print(f"⚠️ Offline load failed: {e}. Trying simple load (if internet is on)...")
            self.encoder = AutoModel.from_pretrained("facebook/dinov2-large")

        # DINO Large = 1024 embedding dim
        self.embed_dim = 1024 
        
        self.decoder = nn.Sequential(
            nn.Conv2d(self.embed_dim, 512, 3, padding=1), 
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Conv2d(512, 256, 3, padding=1), 
            nn.BatchNorm2d(256), nn.ReLU(),
            
            nn.Conv2d(256, 64, 3, padding=1), 
            nn.BatchNorm2d(64), nn.ReLU(),
            
            nn.Conv2d(64, 1, 1)
        )
        
    def forward(self, x):
        outputs = self.encoder(pixel_values=x)
        last_hidden_state = outputs.last_hidden_state
        B, N, C = last_hidden_state.shape
        H_feat = int(math.sqrt(N-1))
        
        features = last_hidden_state[:, 1:, :].permute(0, 2, 1).reshape(B, C, H_feat, H_feat)
        masks = self.decoder(features)
        masks = F.interpolate(masks, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return masks

# ==========================================
# 3. SLIDING WINDOW INFERENCE ENGINE
# ==========================================
def predict_tiled(model, image, tile_size=1024, overlap=0.25):
    h, w, c = image.shape
    stride = int(tile_size * (1 - overlap))
    
    # Add padding to make image divisible by stride
    pad_h = (tile_size - h % stride) % stride
    pad_w = (tile_size - w % stride) % stride
    
    # Check if image is smaller than tile size (pad it)
    if h < tile_size: pad_h += (tile_size - h)
    if w < tile_size: pad_w += (tile_size - w)
        
    image_pad = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    
    h_pad, w_pad, _ = image_pad.shape
    full_mask = torch.zeros((h_pad, w_pad), device=CONFIG.device)
    count_mask = torch.zeros((h_pad, w_pad), device=CONFIG.device)
    
    tiles = []
    coords = []
    
    # 1. Cut Image into Tiles
    for y in range(0, h_pad - tile_size + 1, stride):
        for x in range(0, w_pad - tile_size + 1, stride):
            tiles.append(image_pad[y:y+tile_size, x:x+tile_size])
            coords.append((y, x))
            
    if len(tiles) == 0: # Safety fallback
        tiles.append(cv2.resize(image, (tile_size, tile_size)))
        coords.append((0,0))

    # 2. Predict Batch by Batch
    model.eval()
    with torch.no_grad():
        for i in range(0, len(tiles), CONFIG.batch_size):
            batch_tiles_np = np.array(tiles[i:i+CONFIG.batch_size])
            
            # Normalize manually (ImageNet stats) for speed
            # (Batch, H, W, C) -> (Batch, C, H, W) and normalize
            batch_tiles = torch.from_numpy(batch_tiles_np).float().permute(0,3,1,2).to(CONFIG.device)
            batch_tiles = (batch_tiles / 255.0 - torch.tensor([0.485, 0.456, 0.406], device=CONFIG.device).view(1,3,1,1)) / \
                          torch.tensor([0.229, 0.224, 0.225], device=CONFIG.device).view(1,3,1,1)
            
            # Forward Pass
            preds = torch.sigmoid(model(batch_tiles))
            
            # Test Time Augmentation (Horizontal Flip)
            preds_flip = torch.sigmoid(torch.flip(model(torch.flip(batch_tiles, [3])), [3]))
            preds = (preds + preds_flip) / 2.0
            
            # Stitch into full mask
            for pred, (y, x) in zip(preds, coords[i:i+CONFIG.batch_size]):
                full_mask[y:y+tile_size, x:x+tile_size] += pred.squeeze()
                count_mask[y:y+tile_size, x:x+tile_size] += 1
                
    # Average overlapping areas
    full_mask /= (count_mask + 1e-6)
    
    # Crop back to original size
    return full_mask[:h, :w].cpu().numpy()

def rle_encode(mask):
    pixels = mask.T.flatten()
    dots = np.where(pixels == 1)[0]
    if len(dots) == 0: return "authentic"
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1: run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return json.dumps([int(x) for x in run_lengths])

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    print(f"Device: {CONFIG.device}")
    
    # 1. Load Model
    if not os.path.exists(CONFIG.model_path):
        print(f"❌ CRITICAL ERROR: Model weights not found at {CONFIG.model_path}")
        return

    print("🚀 Loading Model...")
    model = DinoLargeSegmenter()
    
    # Load weights safely (handling DataParallel prefixes)
    checkpoint = torch.load(CONFIG.model_path, map_location=CONFIG.device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # Clean keys if they have "module." prefix from multi-gpu training
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(CONFIG.device)
    model.eval()
    print("✅ Model Loaded Successfully!")

    # 2. Get Test Images
    if os.path.exists(CONFIG.test_images_path):
        image_files = sorted([f for f in os.listdir(CONFIG.test_images_path) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))])
        print(f"Found {len(image_files)} test images.")
    else:
        print("Test directory not found. Exiting.")
        image_files = []

    # 3. Inference Loop
    predictions = []
    print("Starting inference...")
    
    for i, image_name in enumerate(image_files):
        image_path = os.path.join(CONFIG.test_images_path, image_name)
        
        # Load Image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Predict
        prob_map = predict_tiled(model, image, tile_size=CONFIG.tile_size, overlap=CONFIG.tile_overlap)
        
        # Post-process
        mask = (prob_map > CONFIG.min_confidence).astype(np.uint8)
        
        # Morphological Cleanup
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Logic: Authentic vs Forged
        total_pixels = image.shape[0] * image.shape[1]
        if mask.sum() < (total_pixels * CONFIG.min_area):
            rle = "authentic"
        else:
            rle = rle_encode(mask)
            
        predictions.append({
            "case_id": Path(image_name).stem,
            "annotation": rle
        })
        
        if i % 10 == 0:
            print(f"Processed {i+1}/{len(image_files)}")
            gc.collect()

    # 4. Generate Submission File
    print("Generating submission.csv...")
    predictions_df = pd.DataFrame(predictions)
    predictions_df["case_id"] = predictions_df["case_id"].astype(str)
    
    if os.path.exists(CONFIG.sample_sub_path):
        submission = pd.read_csv(CONFIG.sample_sub_path)
        submission["case_id"] = submission["case_id"].astype(str)
        
        # Merge to ensure correct order
        submission = submission[["case_id"]].merge(predictions_df[["case_id", "annotation"]], 
                                                 on="case_id", 
                                                 how="left")
        
        submission["annotation"] = submission["annotation"].fillna("authentic")
        submission[["case_id", "annotation"]].to_csv("submission.csv", index=False)
        print("✅ Submission saved successfully.")
    else:
        predictions_df.to_csv("submission.csv", index=False)
        print("✅ Submission saved (no sample file found).")

if __name__ == "__main__":
    main()