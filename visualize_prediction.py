import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image

# ==========================
# CONFIG
# ==========================
MODEL_PATH = "best_model_fold0.pth"                # trained checkpoint
IMAGE_DIR = "recodai-luc-scientific-image-forgery-detection/test_images"  # folder containing test images
# list of example filenames (adjust or leave empty to auto‑scan directory)
IMAGE_NAMES = [
    "0001.jpg",
    "0002.jpg",
    "0003.jpg",
    "0004.jpg",
    "0005.jpg",
]
OUTPUT_DIR = "visualizations"                        # where side-by-side PNGs will be saved
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# Load Model Architecture
# ==========================
# the checkpoint was trained with binary segmentation (2 output channels),
# so the model must be created with num_classes=2.  later we select the
# second channel as the forgery probability.
model = deeplabv3_resnet50(weights=None, num_classes=2)
model.to(DEVICE)

# ==========================
# Load Checkpoint Correctly
# ==========================
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# extract state dict if wrapped
if isinstance(checkpoint, dict):
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

# remove potential "module." or "model." prefixes
new_state_dict = {}
for k, v in state_dict.items():
    name = k
    if name.startswith("module."):
        name = name[len("module."):]
    if name.startswith("model."):
        name = name[len("model."):]
    new_state_dict[name] = v

model.load_state_dict(new_state_dict, strict=False)
model.eval()
print("Model loaded successfully!")

print("Model loaded successfully!")

# ==========================
# Preprocess helper
# ==========================
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])


# ==========================
# inference loop
# ==========================
# gather image list: either from IMAGE_NAMES or directory scan
if not IMAGE_NAMES:
    # collect first five PNG/JPG files
    IMAGE_NAMES = [f for f in os.listdir(IMAGE_DIR)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:5]

os.makedirs(OUTPUT_DIR, exist_ok=True)

for img_name in IMAGE_NAMES:
    image_path = os.path.join(IMAGE_DIR, img_name)
    if not os.path.exists(image_path):
        print(f"⚠️ {image_path} not found, skipping")
        continue

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # ==========================
    # Inference
    # ==========================
    with torch.no_grad():
        output = model(input_tensor)["out"]        # shape: [1,2,H,W]
        probs = torch.softmax(output, dim=1)[0, 1]  # probability of class=1 (forgery)
        mask = (probs > 0.5).float()

    mask_np = mask.squeeze().cpu().numpy()

    # ==========================
    # Overlay
    # ==========================
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = cv2.resize(original, (1024, 1024))

    overlay = original.copy()
    overlay[mask_np > 0.5] = [255, 0, 0]  # red forgery region (RGB)

    alpha = 0.5
    blended = cv2.addWeighted(original, 1 - alpha, overlay, alpha, 0)

    # Side-by-side
    comparison = np.hstack([original, blended])
    output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_name)[0]}_viz.png")
    cv2.imwrite(output_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

    print(f"Saved visualization to: {output_path}")

