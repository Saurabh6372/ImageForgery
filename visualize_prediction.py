import os
import argparse
import subprocess
import torch
import cv2
import numpy as np
from torchvision import transforms
from torchvision.models import segmentation
from PIL import Image

# ==========================
# CONFIG
# ==========================
MODEL_PATH = "best_model_fold0.pth"                # trained checkpoint
IMAGE_DIR = "recodai-luc-scientific-image-forgery-detection"  # root dataset folder
NUM_IMAGES = None                                  # how many examples to process; None = all test images
OUTPUT_DIR = "visualizations"                     # where side-by-side PNGs will be saved
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# threshold for forgery probability
THRESHOLD = 0.5

# ==========================
# Load Model Architecture
# ==========================
# we allow choosing architecture via command-line (resnet50 or resnet101)

parser = argparse.ArgumentParser(description="Visualize predictions with a segmentation model")
parser.add_argument("--arch", type=str, default="deeplabv3_resnet50",
                    help="model architecture (deeplabv3_resnet50 or deeplabv3_resnet101)")
parser.add_argument("--threshold", type=float, default=THRESHOLD,
                    help="probability threshold for forgery mask")
parser.add_argument("--num", type=int, default=NUM_IMAGES,
                    help="optional: limit number of test images to process")
args = parser.parse_args()
THRESHOLD = args.threshold
NUM_IMAGES = args.num
print(f"Threshold set to {THRESHOLD}")
if NUM_IMAGES is not None:
    print(f"Processing up to {NUM_IMAGES} test images")

# create model dynamically
if args.arch not in ["deeplabv3_resnet50", "deeplabv3_resnet101"]:
    raise ValueError(f"Unsupported architecture {args.arch}")
model_ctor = getattr(segmentation, args.arch)
model = model_ctor(weights=None, num_classes=2)
model.to(DEVICE)
print(f"Using architecture: {args.arch}")

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

# we only look in test_images folder
test_dir = os.path.join(IMAGE_DIR, "test_images")
if os.path.isdir(test_dir):
    image_paths = glob.glob(os.path.join(test_dir, "*.png"))
    image_paths += glob.glob(os.path.join(test_dir, "*.jpg"))
else:
    raise FileNotFoundError("test_images directory not found")

# sort and possibly trim
image_paths = sorted(image_paths)
if NUM_IMAGES is not None:
    image_paths = image_paths[:NUM_IMAGES]
if not image_paths:
    raise FileNotFoundError("No images found in test_images")

print(f"Processing {len(image_paths)} test images:")
for p in image_paths:
    print("  -", p)

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
        mask = (probs > THRESHOLD).float()

    mask_np = mask.squeeze().cpu().numpy()

    # ==========================
    # Overlay
    # ==========================
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = cv2.resize(original, (1024, 1024))

    # also prepare overlay for original side (slightly transparent red)
    overlay_orig = original.copy()
    alpha_in = 0.3
    overlay_orig[mask_np > 0.5] = (np.array([255, 0, 0]) * alpha_in + overlay_orig[mask_np > 0.5] * (1-alpha_in)).astype(np.uint8)

    overlay = original.copy()
    overlay[mask_np > 0.5] = [255, 0, 0]  # red forgery region (RGB)

    alpha = 0.5
    blended = cv2.addWeighted(original, 1 - alpha, overlay, alpha, 0)

    # Side-by-side with labels
    h, w, _ = original.shape
    # show original with faint mask overlay and prediction
    comparison = np.hstack([overlay_orig, blended])

    # annotate titles
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, 'Original + mask overlay', (10, 30), font, 1, (255,255,255), 2, cv2.LINE_AA)
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
    # open the image so you can immediately view it on macOS
    try:
        subprocess.run(["open", output_path])
    except Exception:
        pass


