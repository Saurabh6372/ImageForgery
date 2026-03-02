# ImageForgery Detection Project

This repository contains code used to participate in the **Recod.ai / LUC Scientific Image Forgery Detection** competition. The focus is on segmenting forged regions in scientific images using a variety of deep learning models.

> 📁 **Note:** model weights (`*.pth`, `*.pt`), datasets, and submission `.csv` files are excluded from the repo using `.gitignore`. Only source code is tracked here.

---

## 🚀 Getting Started

1. **Clone the repo**:
   ```bash
   git clone https://github.com/Saurabh6372/ImageForgery.git
   cd ImageForgery
   ```

2. **Install dependencies**:
   ```bash
   python -m venv .venv        # create virtual environment
   source .venv/bin/activate   # macOS / Linux
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Download data and model weights**
   - Download the **Recod.ai dataset** and extract into `./recodai-luc-scientific-image-forgery-detection`.
   - Place your trained model files (`best_model_fold0.pth`, etc.) in the root directory or update paths in config classes.
   - If running on Kaggle, adjust paths accordingly (`../input/...`).

4. **Run scripts**
   - See individual file descriptions below for usage instructions.

---

## 📁 File Overview

| File | Purpose | Key Differences |
|------|---------|-----------------|
| `Forgery_using_DeepLabV3+.py` | Full training + evaluation pipeline using DeepLabV3+ (ResNet backbones). Includes configuration, dataset class, augmentation, training loop, and optional ensembling. | Intended for local training on Mac; lots of configuration options. Produces `.pth` models. |
| `main.py` | Alternate training script using `segmentation_models_pytorch` (U-Net++ with encoders like EfficientNet or SE-ResNeXt). | Simpler than `Forgery_using_DeepLabV3+.py`; focused on training only. |
| `main2.py` | **Inference pipeline** combining two models (custom CNN and DINOv2 segmenter). Includes TTA, post‑processing, and submission generation. | Built for Kaggle-style inference; references sample submission. |
| `main3.py` | High‑resolution DINOv2 large inference with sliding window tiling. More advanced post‑processing and GPU/CPU checks. | Focused on one large DINOv2 model; suitable for T4 GPUs or high‑RAM machines. |
| `generate_submission_only.py` | Lightweight inference script that loads pre‑trained DeepLabV3+ models and outputs `submission.csv`. | No training code; minimal dependencies. Good for quick submissions using existing `.pth` files. |
| `submission_dinov2_optimized.py` | Optimized DINOv2-large inference for Kaggle (robust offline model loading, configurable TTA, device selection). | Intended as the production submission script—handles offline cases gracefully. |
| `submission_ensemble.py` | Ensemble of DINOv2-large and DeepLabV3+ predictions with weighted blending and shared framework. | Demonstrates how to combine models at inference time for higher accuracy. |
| `test.py` | Placeholder/test file. Contains a simple print statement. | Not used in notebooks; just for sanity checks. |
| `requirements.txt` | Python dependencies split into categories with comments. | Already commented to explain purpose of each library. |
| `.gitignore` | Excludes large model files, datasets, CSVs, environment, etc. | Keeps repo lightweight for GitHub. |


### 📝 Notes on Scripts

- **Training vs Inference**: `Forgery_using_DeepLabV3+.py` and `main.py` perform training; the others are inference or submission utilities.
- **Model Types**: Some scripts use `deeplabv3_resnet*` (traditional segmentation), others rely on `facebook/dinov2-*` transformer models.
- **File Paths**: All scripts have configurable paths in their `Config`/`CFG` classes. Update these to match your local or Kaggle environment.
- **Post-Processing**: Most inference scripts include morphological operations, thresholding, and optional TTA to refine predicted masks.


---

## 📚 Additional Information

- The dataset structure expected is the same as the competition: `train_images/`, `train_masks/`, `test_images/`, etc.
- `best_model_fold*.pth` files are example weight filenames; rename them as needed.
- You can use any combination of these scripts depending on whether you want to retrain, run inference locally, or on Kaggle.

---

## ✨ Tips & Tricks

- Adjust `IMG_SIZE` and `BATCH_SIZE` in configs for your hardware (especially on Macs with limited GPU memory).
- When running on Kaggle, prefer the `submission_*.py` scripts which are designed for offline mode.
- Use `git add . && git commit` after modifying scripts; your `.gitignore` protects against accidentally adding data files.

---

## 🎖️ License

Feel free to fork and modify. This code was written for a Kaggle competition and is provided as-is for educational purposes.


---

*Happy hacking!* 🧠🔍
