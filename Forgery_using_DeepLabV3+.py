# Scientific Image Forgery Detection using DeepLabV3+
# Complete pipeline for Kaggle competition

import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet101, deeplabv3_resnet50
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. CONFIGURATION
# ============================================
class Config:
    # Paths - UPDATE THESE FOR YOUR MAC
    DATA_DIR = './recodai-luc-scientific-image-forgery-detection'
    TRAIN_IMG_DIR = f'{DATA_DIR}/train_images'
    TRAIN_MASK_DIR = f'{DATA_DIR}/train_masks'
    TEST_IMG_DIR = f'{DATA_DIR}/test_images'
    SUPP_IMG_DIR = f'{DATA_DIR}/supplemental_images'
    SUPP_MASK_DIR = f'{DATA_DIR}/supplemental_masks'
    
    # Model parameters
    IMG_SIZE = 256  # Adjust based on your Mac's memory
    BATCH_SIZE = 4  # Reduce if memory issues
    NUM_EPOCHS = 2
    LEARNING_RATE = 1e-4
    NUM_FOLDS = 2
    DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'  # Mac M1/M2
    
    # Ensembling
    USE_ENSEMBLE = True
    MODELS_TO_ENSEMBLE = ['deeplabv3_resnet50', 'deeplabv3_resnet101']
    
    # Post-processing
    USE_MORPHOLOGY = True
    KERNEL_SIZE = 5
    
    # Threshold for forgery detection
    THRESHOLD = 0.5

# ============================================
# 2. RLE ENCODING/DECODING (From Competition)
# ============================================
def rle_encode(mask):
    """Convert mask to run-length encoding"""
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape):
    """Decode run-length encoding to mask"""
    if mask_rle == 'authentic':
        return np.zeros(shape, dtype=np.uint8)
    
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# ============================================
# 3. DATASET CLASS
# ============================================
class ForgeryDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None, is_test=False, images_list=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.is_test = is_test
        if images_list is not None:
            self.images = images_list
        else:
            self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.is_test:
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            return image, img_name.replace('.png', '')
        
        # Extract case_id from filename (remove folder and extension)
        case_id = os.path.splitext(os.path.basename(img_name))[0]
        mask_files = [f for f in os.listdir(self.mask_dir) if f.startswith(case_id)]
        
        # Combine multiple masks
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for mask_file in mask_files:
            mask_path = os.path.join(self.mask_dir, mask_file)
            single_mask = np.load(mask_path)
            single_mask = np.squeeze(single_mask)
            if single_mask.ndim > 2:
                single_mask = single_mask[..., 0]
            # Resize to match image size
            if single_mask.shape != image.shape[:2]:
                single_mask = cv2.resize(single_mask.astype(np.float32), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = np.maximum(mask, single_mask.astype(np.uint8))
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.long(), case_id

# ============================================
# 4. AUGMENTATION PIPELINES
# ============================================
def get_train_transforms(img_size):
    """Multi-scale augmentation as per hacker tips"""
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.5),
            A.MedianBlur(blur_limit=3, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.CLAHE(clip_limit=4.0, p=0.5),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], is_check_shapes=False)

def get_val_transforms(img_size):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], is_check_shapes=False)

# ============================================
# 5. MODEL DEFINITION
# ============================================
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=2, backbone='resnet101', pretrained=True):
        super(DeepLabV3Plus, self).__init__()
        if backbone == 'resnet101':
            self.model = deeplabv3_resnet101(pretrained=pretrained, aux_loss=False)
        else:
            self.model = deeplabv3_resnet50(pretrained=pretrained, aux_loss=False)
        
        # Modify classifier for binary segmentation (robustly)
        def _replace_last_conv(module, out_channels):
            try:
                # handle nn.Sequential-like heads
                for i in range(len(module) - 1, -1, -1):
                    if isinstance(module[i], nn.Conv2d):
                        in_ch = module[i].in_channels
                        module[i] = nn.Conv2d(in_ch, out_channels, kernel_size=1)
                        return True
            except Exception:
                # fallback: try to find any Conv2d child
                for name, child in module.named_children():
                    if isinstance(child, nn.Conv2d):
                        in_ch = child.in_channels
                        setattr(module, name, nn.Conv2d(in_ch, out_channels, kernel_size=1))
                        return True
            return False

        if hasattr(self.model, 'classifier') and self.model.classifier is not None:
            _replace_last_conv(self.model.classifier, num_classes)
        if hasattr(self.model, 'aux_classifier') and self.model.aux_classifier is not None:
            _replace_last_conv(self.model.aux_classifier, num_classes)
    
    def forward(self, x):
        return self.model(x)['out']

# ============================================
# 6. LOSS FUNCTIONS
# ============================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)[:, 1]
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, pred, target):
        return 0.5 * self.ce(pred, target) + 0.5 * self.dice(pred, target)

# ============================================
# 7. TRAINING FUNCTION
# ============================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for images, masks, _ in tqdm(loader, desc='Training'):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, masks, _ in tqdm(loader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
    
    return total_loss / len(loader)

# ============================================
# 8. POST-PROCESSING
# ============================================
def post_process_mask(mask, kernel_size=5):
    """Apply morphological operations to clean masks"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

# ============================================
# 9. INFERENCE WITH TTA (Test Time Augmentation)
# ============================================
def predict_with_tta(model, image, device, tta_transforms=None):
    """Predict with test-time augmentation"""
    model.eval()
    predictions = []
    
    # Original prediction
    with torch.no_grad():
        img_tensor = image.unsqueeze(0).to(device)
        pred = torch.softmax(model(img_tensor), dim=1)[0, 1].cpu().numpy()
        predictions.append(pred)
    
    # TTA transforms
    if tta_transforms:
        # Horizontal flip
        img_flip = torch.flip(image, [2])
        with torch.no_grad():
            img_tensor = img_flip.unsqueeze(0).to(device)
            pred = torch.softmax(model(img_tensor), dim=1)[0, 1].cpu().numpy()
            pred = np.fliplr(pred)
            predictions.append(pred)
        
        # Vertical flip
        img_flip = torch.flip(image, [1])
        with torch.no_grad():
            img_tensor = img_flip.unsqueeze(0).to(device)
            pred = torch.softmax(model(img_tensor), dim=1)[0, 1].cpu().numpy()
            pred = np.flipud(pred)
            predictions.append(pred)
    
    return np.mean(predictions, axis=0)

# ============================================
# 10. MAIN TRAINING PIPELINE WITH CROSS-VALIDATION
# ============================================
def train_model_cv(config):
    """Train model with K-Fold cross-validation"""
    
    # Prepare data - collect all images from subfolders
    all_images = []
    for root, dirs, files in os.walk(config.TRAIN_IMG_DIR):
        for file in files:
            if file.endswith('.png'):
                # Store relative path from TRAIN_IMG_DIR
                rel_path = os.path.relpath(os.path.join(root, file), config.TRAIN_IMG_DIR)
                all_images.append(rel_path)
    all_images = sorted(all_images)
    
    kfold = KFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=42)
    
    fold_models = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_images)):
        print(f"\n{'='*50}")
        print(f"Training Fold {fold + 1}/{config.NUM_FOLDS}")
        print(f"{'='*50}")
        
        # Get train/val image lists for this fold
        train_images = [all_images[i] for i in train_idx]
        val_images = [all_images[i] for i in val_idx]
        
        # Create datasets
        train_dataset = ForgeryDataset(
            config.TRAIN_IMG_DIR,
            config.TRAIN_MASK_DIR,
            transform=get_train_transforms(config.IMG_SIZE),
            images_list=train_images
        )
        
        val_dataset = ForgeryDataset(
            config.TRAIN_IMG_DIR,
            config.TRAIN_MASK_DIR,
            transform=get_val_transforms(config.IMG_SIZE),
            images_list=val_images
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                                 shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                               shuffle=False, num_workers=2)
        
        # Initialize model
        model = DeepLabV3Plus(num_classes=2, backbone='resnet101', pretrained=True)
        model = model.to(config.DEVICE)
        
        criterion = CombinedLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.NUM_EPOCHS)
        
        best_val_loss = float('inf')
        
        for epoch in range(config.NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
            
            train_loss = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
            val_loss = validate_epoch(model, val_loader, criterion, config.DEVICE)
            scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'best_model_fold{fold}.pth')
                print(f"Model saved! Best Val Loss: {best_val_loss:.4f}")
        
        fold_models.append(f'best_model_fold{fold}.pth')
    
    return fold_models

# ============================================
# 11. PREDICTION & SUBMISSION
# ============================================
def generate_submission(config, model_paths):
    """Generate submission file with ensemble predictions"""
    
    # Load models
    models = []
    for path in model_paths:
        model = DeepLabV3Plus(num_classes=2, backbone='resnet101', pretrained=False)
        model.load_state_dict(torch.load(path, map_location=config.DEVICE), strict=False)
        model = model.to(config.DEVICE)
        model.eval()
        models.append(model)
    
    # Prepare test dataset
    test_dataset = ForgeryDataset(
        config.TEST_IMG_DIR,
        transform=get_val_transforms(config.IMG_SIZE),
        is_test=True
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Generate predictions
    submissions = []
    
    for image, case_id in tqdm(test_loader, desc='Generating predictions'):
        image = image.to(config.DEVICE)
        
        # Ensemble predictions
        ensemble_pred = []
        for model in models:
            pred = predict_with_tta(model, image[0], config.DEVICE, tta_transforms=True)
            ensemble_pred.append(pred)
        
        # Average ensemble predictions
        final_pred = np.mean(ensemble_pred, axis=0)
        
        # Threshold
        mask = (final_pred > config.THRESHOLD).astype(np.uint8)
        
        # Resize to original size
        original_img = cv2.imread(os.path.join(config.TEST_IMG_DIR, f"{case_id[0]}.png"))
        mask = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]))
        
        # Post-process
        if config.USE_MORPHOLOGY:
            mask = post_process_mask(mask, config.KERNEL_SIZE)
        
        # Encode
        if mask.sum() == 0:
            annotation = 'authentic'
        else:
            annotation = rle_encode(mask)
        
        submissions.append({
            'case_id': case_id[0],
            'annotation': annotation
        })
    
    # Create submission file
    submission_df = pd.DataFrame(submissions)
    submission_df.to_csv('submission.csv', index=False)
    print("\nSubmission file created: submission.csv")

# ============================================
# 12. MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    print("="*70)
    print("Scientific Image Forgery Detection Pipeline")
    print("="*70)
    
    config = Config()
    
    print(f"\nDevice: {config.DEVICE}")
    print(f"Image Size: {config.IMG_SIZE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    
    # Train models with cross-validation
    print("\n" + "="*70)
    print("PHASE 1: Training with Cross-Validation")
    print("="*70)
    model_paths = train_model_cv(config)
    
    # Generate submission
    print("\n" + "="*70)
    print("PHASE 2: Generating Submission")
    print("="*70)
    generate_submission(config, model_paths)
    
    print("\n" + "="*70)
    print("Pipeline Complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Review submission.csv")
    print("2. Upload to Kaggle")
    print("3. Check leaderboard score")
    print("4. Iterate and improve!")