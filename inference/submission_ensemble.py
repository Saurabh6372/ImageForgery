import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
from torchvision.models.segmentation import deeplabv3_resnet101
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm

# ==========================================
# ⚙️ CONFIG
# ==========================================
class CFG:
    # --- Data Paths (Robust) ---
    DATA_DIR = None
    POSSIBLE_DIRS = [
        './recodai-luc-scientific-image-forgery-detection',
        '../input/recodai-luc-scientific-image-forgery-detection',
        'dataset',
        './' 
    ]
    for d in POSSIBLE_DIRS:
        if os.path.exists(os.path.join(d, 'test_images')):
            DATA_DIR = d
            break
    if DATA_DIR is None:
        # Fallback if allowed, otherwise script might fail later, but we try standard
        DATA_DIR = './' 
        print("⚠️ Warning: Could not find dataset folder. Assuming current directory has test_images/")
        
    TEST_IMG_DIR = f'{DATA_DIR}/test_images'
    
    # --- Device ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # --- Ensemble Settings ---
    # Weightage for blending: DINO is stronger, so 0.6 vs 0.4
    WEIGHT_DINO = 0.6
    WEIGHT_DEEPLAB = 0.4
    
    THRESHOLD = 0.5
    USE_TTA = True # Flip augmentation
    USE_MORPHOLOGY = True
    KERNEL_SIZE = 5

    # --- Model 1: DINOv2 ---
    DINO_MODEL_NAME = "facebook/dinov2-large"
    # Matches Main3.py / training notebook
    DINO_WEIGHTS = "/kaggle/input/dinoexteme/other/default/1/dinov2_large_a40_extreme.pth" 
    DINO_IMG_SIZE = 1024
    
    # --- Model 2: DeepLabV3+ ---
    # Matches generate_submission_only.py
    DEEPLAB_WEIGHTS = [
        "best_model_fold0.pth",
        "best_model_fold1.pth"
    ]
    DEEPLAB_IMG_SIZE = 256

# ==========================================
# 🧠 MODEL 1: DINOv2 (High Res)
# ==========================================
class DinoLargeSegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Robust Offline Loading
        base_model_path = CFG.DINO_MODEL_NAME
        offline_paths = [
            "/kaggle/input/dinov2/pytorch/large/1", # Correct one per recent finding
            "/kaggle/input/facebook-dinov2-large",
            "/kaggle/input/dinov2-large",
            "../input/facebook-dinov2-large", 
            "../input/dinov2-large"
        ]
        
        for p in offline_paths:
            if os.path.exists(p) and os.path.exists(os.path.join(p, "config.json")):
                print(f"✅ Found local DINO base: {p}")
                base_model_path = p
                break
        
        try:
            self.encoder = AutoModel.from_pretrained(base_model_path)
        except Exception as e:
            print(f"⚠️ Error loading DINO base: {e}")
            # Diagnostic for Kaggle
            print("Did you add the facebook/dinov2-large dataset?")
            raise
            
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
        masks = F.interpolate(masks, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return masks

# ==========================================
# 🧠 MODEL 2: DeepLabV3+ (Low Res)
# ==========================================
class DeepLabV3Binary(nn.Module):
    def __init__(self):
        super().__init__()
        # Pretrained=False because we load custom weights
        self.model = deeplabv3_resnet101(pretrained=False)
        # Update classifier for binary class
        self.model.classifier[-1] = nn.Conv2d(256, 2, kernel_size=1)
        if self.model.aux_classifier is not None:
            self.model.aux_classifier[-1] = nn.Conv2d(256, 2, kernel_size=1)

    def forward(self, x):
        return self.model(x)['out']

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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# ==========================================
# 📂 DATASET
# ==========================================
class EnsembleDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.files = sorted([f for f in os.listdir(directory) if f.endswith('.png')])
        
        # Transforms for each model
        self.tf_dino = A.Compose([
            A.Resize(CFG.DINO_IMG_SIZE, CFG.DINO_IMG_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        self.tf_deeplab = A.Compose([
            A.Resize(CFG.DEEPLAB_IMG_SIZE, CFG.DEEPLAB_IMG_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        filename = self.files[idx]
        path = os.path.join(self.directory, filename)
        
        # Read Original
        img_orig = cv2.imread(path)
        img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        
        # Prepare for DINO
        img_dino = self.tf_dino(image=img_rgb)['image']
        
        # Prepare for DeepLab
        img_deeplab = self.tf_deeplab(image=img_rgb)['image']
        
        case_id = os.path.splitext(filename)[0]
        
        return {
            'dino': img_dino,
            'deeplab': img_deeplab,
            'case_id': case_id,
            'orig_size': img_orig.shape[:2] # (H, W)
        }

# ==========================================
# 🚀 MAIN
# ==========================================
def generate_submission():
    print(f"🔥 Starting Ensemble Inference on {CFG.DEVICE}")
    
    # 1. Load DINOv2
    print("--- Loading DINOv2 ---")
    dino = DinoLargeSegmenter()
    # Check if weights exist (locally or kaggle)
    dino_w_path = CFG.DINO_WEIGHTS
    if not os.path.exists(dino_w_path) and os.path.exists("dinov2_large_a40_extreme (1).pth"):
        dino_w_path = "dinov2_large_a40_extreme (1).pth"
        
    if os.path.exists(dino_w_path):
        state_dict = torch.load(dino_w_path, map_location=CFG.DEVICE)
        # Fix DataParallel keys
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        dino.load_state_dict(new_state_dict)
        dino.to(CFG.DEVICE)
        dino.eval()
        print(f"✅ DINO Loaded from {dino_w_path}")
    else:
        print(f"❌ DINO Weights not found at {dino_w_path}. Check path!")
        return

    # 2. Load DeepLabs
    print("--- Loading DeepLabs ---")
    deeplabs = []
    for w_name in CFG.DEEPLAB_WEIGHTS:
        if os.path.exists(w_name):
            dl = DeepLabV3Binary()
            dl.load_state_dict(torch.load(w_name, map_location=CFG.DEVICE), strict=False)
            dl.to(CFG.DEVICE)
            dl.eval()
            deeplabs.append(dl)
            print(f"✅ DeepLab Loaded: {w_name}")
        else:
             # Try /kaggle/input search if not in current dir
             # Assuming user might have them in a dataset
             print(f"⚠️ DeepLab Weights {w_name} not found locally.")

    if not deeplabs:
        print("⚠️ No DeepLab models found! Falling back to DINO only.")
        CFG.WEIGHT_DINO = 1.0
        CFG.WEIGHT_DEEPLAB = 0.0

    # 3. Predict
    dataset = EnsembleDataset(CFG.TEST_IMG_DIR)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    
    submission = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Ensembling"):
            case_id = batch['case_id'][0]
            orig_h, orig_w = batch['orig_size'][0].item(), batch['orig_size'][1].item()
            
            # --- DINO Prediction ---
            img_d = batch['dino'].to(CFG.DEVICE)
            
            # TTA DINO
            preds_d = []
            preds_d.append(torch.sigmoid(dino(img_d)).cpu().numpy())
            if CFG.USE_TTA:
                preds_d.append(torch.flip(torch.sigmoid(dino(torch.flip(img_d, [3]))), [3]).cpu().numpy()) # H-Flip
                preds_d.append(torch.flip(torch.sigmoid(dino(torch.flip(img_d, [2]))), [2]).cpu().numpy()) # V-Flip
            
            # Average DINO
            dino_map = np.mean(preds_d, axis=0)[0, 0] # (1024, 1024)
            # Resize DINO to Original
            dino_map = cv2.resize(dino_map, (orig_w, orig_h))
            
            # --- DeepLab Prediction ---
            deeplab_map = np.zeros_like(dino_map)
            
            if deeplabs:
                img_dl = batch['deeplab'].to(CFG.DEVICE)
                preds_dl = []
                for dl_model in deeplabs:
                    # Model outputs logits for 2 classes. We want softmax of class 1.
                    out = torch.softmax(dl_model(img_dl), dim=1)[:, 1].unsqueeze(1)
                    preds_dl.append(out.cpu().numpy())
                    
                    if CFG.USE_TTA:
                        out_h = torch.softmax(dl_model(torch.flip(img_dl, [3])), dim=1)[:, 1].unsqueeze(1)
                        preds_dl.append(torch.flip(out_h, [3]).cpu().numpy())
                        
                # Average all DeepLabs
                dl_avg = np.mean(preds_dl, axis=0)[0, 0] # (256, 256)
                # Resize to Original
                deeplab_map = cv2.resize(dl_avg, (orig_w, orig_h))
            
            # --- Ensemble ---
            final_map = (dino_map * CFG.WEIGHT_DINO) + (deeplab_map * CFG.WEIGHT_DEEPLAB)
            
            # Validate shapes/types
            mask_bin = (final_map > CFG.THRESHOLD).astype(np.uint8)
            
            # Morphology
            if CFG.USE_MORPHOLOGY:
                mask_bin = post_process_mask(mask_bin, CFG.KERNEL_SIZE)
                
            # Encode
            if mask_bin.sum() == 0:
                rle = 'authentic'
            else:
                rle = rle_encode(mask_bin)
            
            submission.append({
                'case_id': case_id,
                'annotation': rle
            })
            
    df = pd.DataFrame(submission)
    df.to_csv('submission.csv', index=False)
    print("✅ Ensemble submission.csv generated!")

if __name__ == "__main__":
    generate_submission()
