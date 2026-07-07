import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm

# ==========================================
# ⚙️ CONFIG (Updated for DINOv2 High Res)
# ==========================================
class CFG:
    # Paths - UPDATE THESE FOR KAGGLE
    # Paths - Robus Search
    DATA_DIR = None
    POSSIBLE_DIRS = [
        './recodai-luc-scientific-image-forgery-detection',
        '../input/recodai-luc-scientific-image-forgery-detection',
        'dataset',
        './' # Fallback
    ]
    
    for d in POSSIBLE_DIRS:
        if os.path.exists(os.path.join(d, 'test_images')):
            DATA_DIR = d
            break
            
    if DATA_DIR is None:
        raise FileNotFoundError("Could not find dataset directory!")
        
    TEST_IMG_DIR = f'{DATA_DIR}/test_images'
    print(f"📂 Using Data Directory: {DATA_DIR}")
    
    # Model
    MODEL_NAME = "facebook/dinov2-large"
    WEIGHTS_PATH = "/kaggle/input/dinoexteme/other/default/1/dinov2_large_a40_extreme.pth"
    
    # Inference Settings
    IMG_SIZE = 1024        # MATCH TRAINING SIZE!
    BATCH_SIZE = 1         # Safe for inference
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Post-processing
    THRESHOLD = 0.5
    USE_TTA = True         # Test Time Augmentation
    USE_MORPHOLOGY = True
    KERNEL_SIZE = 5

# ==========================================
# 🧠 MODEL ARCHITECTURE (MUST MATCH TRAINING)
# ==========================================
class DinoLargeSegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Try Loading Base Model (Offline Support)
        base_model_path = CFG.MODEL_NAME
        
        # Search for local copy of the model (For Kaggle Offline)
        # Common locations where users/Kaggle might put the dataset
        offline_paths = [
            "/kaggle/input/dinov2/pytorch/large/1", # User provided path
            "/kaggle/input/facebook-dinov2-large",
            "/kaggle/input/dinov2-large",
            "../input/facebook-dinov2-large", 
            "../input/dinov2-large"
        ]
        
        # Check if we are potentially offline or if the hub is unreachable
        # We try to find a local path first if it exists
        for p in offline_paths:
            if os.path.exists(p) and os.path.exists(os.path.join(p, "config.json")):
                print(f"✅ Found local base model at: {p}")
                base_model_path = p
                break
        
        try:
            self.encoder = AutoModel.from_pretrained(base_model_path)
            print(f"✅ Loaded base model from: {base_model_path}")
        except (OSError, ValueError):
            # If we are here, it means we couldn't connect to HuggingFace AND didn't find a local copy.
            print("\n❌ CRITICAL ERROR: Could not load base model `facebook/dinov2-large`.")
            print("   Since Internet is disabled, you MUST add the model as a Kaggle Dataset.")
            
            # DIAGNOSTIC: List available datasets to help user find the right one
            print("\n🔎 DIAGNOSTIC: Available Datasets in /kaggle/input:")
            try:
                for root, dirs, files in os.walk("/kaggle/input"):
                    for d in dirs:
                        print(f"   - {os.path.join(root, d)}")
                    # Limit depth
                    break 
            except:
                print("   (Could not list directories)")
                
            print("\n👉 ACTION REQUIRED:")
            print("   1. If you see your dino dataset above, update `offline_paths` in the code to include it.")
            print("   2. If not, click 'Add Data' -> Search 'facebook/dinov2-large' -> Add it.")
            print("   3. Then run this script again.")
            raise RuntimeError("Base model not found in offline mode.")

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
        H_feat = int(np.sqrt(N-1))
        
        features = last_hidden_state[:, 1:, :].permute(0, 2, 1).reshape(B, C, H_feat, H_feat)
        masks = self.decoder(features)
        
        # Up-sample to input size
        masks = F.interpolate(masks, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return masks

# ==========================================
# 🛠 UTILS
# ==========================================
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def post_process_mask(mask, k=5):
    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close gaps
    return mask

# ==========================================
# 📂 DATASET
# ==========================================
class TestDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.files = sorted([f for f in os.listdir(directory) if f.endswith('.png')])
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        filename = self.files[idx]
        path = os.path.join(self.directory, filename)
        
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
            
        case_id = os.path.splitext(filename)[0]
        return img, case_id

def get_transforms():
    return A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# ==========================================
# 🚀 MAIN INFERENCE LOOP
# ==========================================
def generate_submission():
    print(f"🔥 Starting Inference on {CFG.DEVICE}")
    print(f"   Model: {CFG.MODEL_NAME}")
    print(f"   Weights: {CFG.WEIGHTS_PATH}")
    print(f"   Image Size: {CFG.IMG_SIZE}")
    
    # 1. Load Model
    model = DinoLargeSegmenter()
    
    # Handle DataParallel vs Single GPU saving
    state_dict = torch.load(CFG.WEIGHTS_PATH, map_location=CFG.DEVICE)
    # Fix keys if they were saved with "module." prefix (DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(CFG.DEVICE)
    model.eval()
    
    # 2. Setup Data
    test_ds = TestDataset(CFG.TEST_IMG_DIR, transform=get_transforms())
    test_loader = DataLoader(test_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=2)
    
    submission_data = []
    
    # 3. Predict
    with torch.no_grad():
        for imgs, case_ids in tqdm(test_loader, desc="Predicting"):
            imgs = imgs.to(CFG.DEVICE)
            
            # --- TTA (Test Time Augmentation) ---
            preds = []
            
            # Current
            out = model(imgs)
            pred_orig = torch.sigmoid(out)
            preds.append(pred_orig.cpu().numpy())
            
            if CFG.USE_TTA:
                # Horizontal Flip
                imgs_h = torch.flip(imgs, [3])
                out_h = model(imgs_h)
                pred_h = torch.flip(torch.sigmoid(out_h), [3])
                preds.append(pred_h.cpu().numpy())
                
                # Vertical Flip
                imgs_v = torch.flip(imgs, [2])
                out_v = model(imgs_v)
                pred_v = torch.flip(torch.sigmoid(out_v), [2])
                preds.append(pred_v.cpu().numpy())
            
            # Average Predictions
            final_pred = np.mean(preds, axis=0) # (B, 1, H, W)
            
            # Process each image in batch
            for i in range(len(case_ids)):
                # Get single mask (H, W)
                mask_prob = final_pred[i, 0]
                
                # Threshold
                mask_bin = (mask_prob > CFG.THRESHOLD).astype(np.uint8)
                
                # Resize back to original size (IMPORTANT!)
                # We need to read the original file to know the size, or assume standard?
                # Best practice: Read original dimensions.
                orig_path = os.path.join(CFG.TEST_IMG_DIR, case_ids[i] + ".png")
                orig_img = cv2.imread(orig_path)
                h_orig, w_orig = orig_img.shape[:2]
                
                if mask_bin.shape[:2] != (h_orig, w_orig):
                    mask_bin = cv2.resize(mask_bin, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                
                # Morphology
                if CFG.USE_MORPHOLOGY:
                    mask_bin = post_process_mask(mask_bin, CFG.KERNEL_SIZE)
                
                # Encode
                if mask_bin.sum() == 0:
                    rle = 'authentic'
                else:
                    rle = rle_encode(mask_bin)
                    
                submission_data.append({
                    "case_id": case_ids[i],
                    "annotation": rle
                })
                
    # 4. Save CSV
    df = pd.DataFrame(submission_data)
    df.to_csv("submission.csv", index=False)
    print("✅ Submission saved to submission.csv")

if __name__ == "__main__":
    generate_submission()
