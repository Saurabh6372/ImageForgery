# ============================================
# Inference-only script for Kaggle submission
# Uses trained DeepLabV3 models (NO TRAINING)
# ============================================

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet101
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ============================================
# CONFIG
# ============================================
class Config:
    DATA_DIR = './recodai-luc-scientific-image-forgery-detection'
    TEST_IMG_DIR = f'{DATA_DIR}/test_images'

    IMG_SIZE = 256
    DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

    THRESHOLD = 0.5
    USE_MORPHOLOGY = True
    KERNEL_SIZE = 5

    # 🔥 YOUR TRAINED MODELS
    MODEL_PATHS = [
        'best_model_fold0.pth',
        'best_model_fold1.pth'
    ]

# ============================================
# RLE ENCODING
# ============================================
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(map(str, runs))

# ============================================
# DATASET (TEST ONLY)
# ============================================
class TestDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=image)
        image = augmented['image']

        case_id = img_name.replace('.png', '')
        return image, case_id

# ============================================
# TRANSFORMS
# ============================================
def get_test_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ============================================
# MODEL
# ============================================
class DeepLabV3Binary(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = deeplabv3_resnet101(pretrained=False)

        # Replace classifier safely
        self.model.classifier[-1] = nn.Conv2d(256, 2, kernel_size=1)

        if self.model.aux_classifier is not None:
            self.model.aux_classifier[-1] = nn.Conv2d(256, 2, kernel_size=1)

    def forward(self, x):
        return self.model(x)['out']

# ============================================
# POST-PROCESSING
# ============================================
def post_process(mask, k=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# ============================================
# TTA PREDICTION
# ============================================
def predict_tta(model, image, device):
    preds = []

    with torch.no_grad():
        out = torch.softmax(model(image.unsqueeze(0).to(device)), dim=1)[0, 1]
        preds.append(out.cpu().numpy())

        # Horizontal flip
        img_h = torch.flip(image, [2])
        out = torch.softmax(model(img_h.unsqueeze(0).to(device)), dim=1)[0, 1]
        preds.append(np.fliplr(out.cpu().numpy()))

        # Vertical flip
        img_v = torch.flip(image, [1])
        out = torch.softmax(model(img_v.unsqueeze(0).to(device)), dim=1)[0, 1]
        preds.append(np.flipud(out.cpu().numpy()))

    return np.mean(preds, axis=0)

# ============================================
# MAIN SUBMISSION GENERATION
# ============================================
def generate_submission():
    cfg = Config()

    print(f"Using device: {cfg.DEVICE}")

    # Load models
    models = []
    for path in cfg.MODEL_PATHS:
        model = DeepLabV3Binary()
        model.load_state_dict(
        torch.load(path, map_location=cfg.DEVICE),
        strict=False
        )

        model.to(cfg.DEVICE)
        model.eval()
        models.append(model)

    # Dataset
    dataset = TestDataset(cfg.TEST_IMG_DIR, get_test_transforms(cfg.IMG_SIZE))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    submissions = []

    for image, case_id in tqdm(loader, desc="Generating submission"):
        image = image[0]

        preds = []
        for model in models:
            preds.append(predict_tta(model, image, cfg.DEVICE))

        final_pred = np.mean(preds, axis=0)
        mask = (final_pred > cfg.THRESHOLD).astype(np.uint8)

        # Resize to original image size
        orig = cv2.imread(os.path.join(cfg.TEST_IMG_DIR, f"{case_id[0]}.png"))
        mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]))

        if cfg.USE_MORPHOLOGY:
            mask = post_process(mask, cfg.KERNEL_SIZE)

        if mask.sum() == 0:
            annotation = 'authentic'
        else:
            annotation = rle_encode(mask)

        submissions.append({
            'case_id': case_id[0],
            'annotation': annotation
        })

    df = pd.DataFrame(submissions)
    df.to_csv('submission.csv', index=False)
    print("\n✅ submission.csv generated successfully!")

# ============================================
# RUN
# ============================================
if __name__ == "__main__":
    generate_submission()
