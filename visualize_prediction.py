import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
from PIL import Image
import matplotlib.pyplot as plt

# You can switch the architecture below by commenting/uncommenting.
# Example: use resnet101 by setting MODEL_ARCH='resnet101'.

# ==========================
# CONFIG
# ==========================
MODEL_PATH = "best_model_fold0.pth"                # trained checkpoint
IMAGE_DIR = "recodai-luc-scientific-image-forgery-detection"  # root dataset folder
NUM_IMAGES = 5                                     # how many examples to process
OUTPUT_DIR = "visualizations"                     # where side-by-side PNGs will be saved
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# Load Model Architecture
# ==========================
# the checkpoint was trained with binary segmentation (2 output channels),
# so the model must be created with num_classes=2.  later we select the
# second channel as the forgery probability.
# change MODEL_ARCH if you want to experiment with different backbones.
MODEL_ARCH = 'resnet50'  # options: 'resnet50', 'resnet101'
if MODEL_ARCH == 'resnet50':
    model = deeplabv3_resnet50(weights=None, num_classes=2)
else:
    model = deeplabv3_resnet101(weights=None, num_classes=2)
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
# collect a few images automatically; prefer test set, fallback to training images
import glob

image_paths = []
# look for test images
test_dir = os.path.join(IMAGE_DIR, "test_images")
if os.path.isdir(test_dir):
    image_paths = glob.glob(os.path.join(test_dir, "*.png"))
    image_paths += glob.glob(os.path.join(test_dir, "*.jpg"))

# if no test images found, search train_images subfolders recursively
if not image_paths:
    train_dir = os.path.join(IMAGE_DIR, "train_images")
    if os.path.isdir(train_dir):
        image_paths = glob.glob(os.path.join(train_dir, "**", "*.png"), recursive=True)
        image_paths += glob.glob(os.path.join(train_dir, "**", "*.jpg"), recursive=True)

# sort and trim
image_paths = sorted(image_paths)[:NUM_IMAGES]
if not image_paths:
    raise FileNotFoundError("No images found in test_images or train_images")

os.makedirs(OUTPUT_DIR, exist_ok=True)

for image_path in image_paths:
    img_name = os.path.basename(image_path)
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

    # Side-by-side with labels
    h, w, _ = original.shape
    comparison = np.hstack([original, blended])

    # annotate titles
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, 'Original', (10, 30), font, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(comparison, 'Prediction (red = forgery)', (w+10, 30), font, 1, (255,255,255), 2, cv2.LINE_AA)

    # optionally draw bounding box around masked region on prediction side
    mask_uint8 = (mask_np > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,ww,hh = cv2.boundingRect(cnt)
        cv2.rectangle(comparison, (w+x, y), (w+x+ww, y+hh), (0,0,255), 2)
        cv2.putText(comparison, 'Forged area', (w+x, y-10), font, 0.6, (0,0,255), 2, cv2.LINE_AA)

    output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_name)[0]}_viz.png")
    cv2.imwrite(output_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

    print(f"Saved visualization to: {output_path}")

    # display using matplotlib for immediate visual feedback
    plt.figure(figsize=(12, 6))
    plt.imshow(comparison)
    plt.axis('off')
    plt.title(f"Original vs Prediction ({img_name})")
    plt.show()
    # try opening in default viewer (macOS)
    try:
        import subprocess
        subprocess.run(["open", output_path])
    except Exception:
        pass


