import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ==========================================
# ⚙️ CONFIGURATION (MAC M4 PRO)
# ==========================================
class CFG:
    # "mps" is the acceleration key for Apple Silicon
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    img_size = 512
    batch_size = 8  # M4 Pro has shared RAM, 16 is safe. Try 32 if you have >32GB RAM.
    epochs = 10      # You have unlimited time, so train longer!
    lr = 2e-4
    num_workers = 2  # Mac handles workers well
    
    # Path to where you downloaded the data on your Mac
    # UPDATE THIS TO YOUR LOCAL FOLDER!
    BASE_PATH = "./recodai-luc-scientific-image-forgery-detection"

    models_to_train = ['efficientnet-b4', 'se_resnext50_32x4d']

# ==========================================
# 🧠 DATASET CLASS
# ==========================================
class ForgeryDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image']
        mask_path = row['mask']
        
        # Load Image
        image = cv2.imread(image_path)
        if image is None: # Safety check
            image = np.zeros((CFG.img_size, CFG.img_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (CFG.img_size, CFG.img_size))
            
        # Load Mask
        if mask_path is not None and str(mask_path) != 'nan':
            try:
                if mask_path.endswith('.npy'):
                    mask = np.load(mask_path)
                    if mask.ndim > 2: mask = np.max(mask, axis=-1)
                else:
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                # Resize and Threshold
                if mask is not None:
                    if mask.shape[:2] != (CFG.img_size, CFG.img_size):
                        mask = cv2.resize(mask, (CFG.img_size, CFG.img_size), interpolation=cv2.INTER_NEAREST)
                    mask = (mask > 0).astype(np.float32)
                else:
                    mask = np.zeros((CFG.img_size, CFG.img_size), dtype=np.float32)
            except:
                mask = np.zeros((CFG.img_size, CFG.img_size), dtype=np.float32)
        else:
            mask = np.zeros((CFG.img_size, CFG.img_size), dtype=np.float32)
            
        # Augmentations
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']
            
        return image, mask.unsqueeze(0)

# ==========================================
# 🏋️ TRAINING LOOP
# ==========================================
def train_model():
    print(f"🚀 Training on: {CFG.device}")
    
    # 1. Prepare Data List
    print("Scanning files...")
    # Update these globs if your local folder structure is slightly different
    train_files = glob(f"{CFG.BASE_PATH}/train_images/*/*.png") + \
                  glob(f"{CFG.BASE_PATH}/supplemental_images/*.png")
    
    data = []
    for p in train_files:
        case_id = os.path.basename(p).split('.')[0]
        # Look for mask in local folders
        mask_path = None
        possible_paths = [
            f"{CFG.BASE_PATH}/train_masks/{case_id}.png",
            f"{CFG.BASE_PATH}/train_masks/{case_id}.npy",
            f"{CFG.BASE_PATH}/supplemental_masks/{case_id}.png"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                mask_path = path
                break
        data.append({'image': p, 'mask': mask_path, 'case_id': case_id})
        
    df = pd.DataFrame(data)
    print(f"Found {len(df)} training images.")
    
    # 2. Dataloader
    transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=90, p=0.5),
        A.Normalize(),
        ToTensorV2()
    ])
    
    dataset = ForgeryDataset(df, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)
    
    # 3. Train Models
    for name in CFG.models_to_train:
        print(f"\n--- Training {name} ---")
        model = smp.UnetPlusPlus(encoder_name=name, encoder_weights="imagenet", in_channels=3, classes=1)
        model.to(CFG.device)
        
        optimizer = optim.AdamW(model.parameters(), lr=CFG.lr)
        loss_fn_dice = smp.losses.DiceLoss(mode='binary')
        loss_fn_bce = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.1)
        
        for epoch in range(CFG.epochs):
            model.train()
            running_loss = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CFG.epochs}")
            for images, masks in pbar:
                images = images.to(CFG.device)
                masks = masks.to(CFG.device)
                
                optimizer.zero_grad()
                
                # Standard Forward Pass (No AMP on MPS to be safe)
                outputs = model(images)
                loss = 0.5 * loss_fn_dice(outputs, masks) + 0.5 * loss_fn_bce(outputs, masks)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
        
        # Save Model
        save_name = f"model_{name}_mac.pth"
        torch.save(model.state_dict(), save_name)
        print(f"✅ Saved {save_name}")

if __name__ == '__main__':
    train_model()