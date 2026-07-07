# Scientific Image Forgery Detection

Pixel-level segmentation of manipulated regions in scientific publication images, built for the **Recod.ai / LUC Scientific Image Forgery Detection** Kaggle competition.

The interesting part of this project is the **systematic model comparison**: classic CNN encoder–decoders (DeepLabV3+, U-Net++) against a **DINOv2-Large vision transformer** used as a segmentation encoder — evaluated on F1, precision/recall trade-offs, and inference latency.

<!-- Add an example here: input image → predicted forgery mask overlay (docs/example.png) -->

## Method

- **Task:** binary pixel classification (authentic vs. forged region), trained on the competition's labelled masks.
- **Models compared:** DeepLabV3+ (ResNet-50/101), U-Net++ (EfficientNet / SE-ResNeXt encoders via `segmentation_models_pytorch`), DINOv2-Large ViT segmentation head.
- **High-resolution inference:** 1024×1024 sliding-window tiling with Gaussian-weighted overlap stitching, so full-resolution masks are reconstructed without boundary artefacts or GPU memory overflow.
- **Post-processing:** Otsu thresholding + morphological filtering (dilation, erosion, connected-component cleanup); thresholds chosen by grid search on the validation split.
- **Ensembling:** weighted blending of DINOv2 and DeepLabV3+ predictions, plus optional test-time augmentation.

**Finding:** DINOv2-Large achieved the best F1 but at ~3× the inference time of the CNN baselines — the CNN/ViT trade-off is documented in the training scripts.

## Repository structure

```
training/
  deeplabv3plus.py        # full DeepLabV3+ training + evaluation pipeline
  unetpp_smp.py           # U-Net++ training via segmentation_models_pytorch
inference/
  ensemble_cnn_dinov2.py        # two-model inference with TTA + post-processing
  dinov2_sliding_window.py      # high-res DINOv2-Large sliding-window inference
  generate_submission.py        # lightweight submission from trained DeepLabV3+ weights
  submission_dinov2_optimized.py  # Kaggle-ready offline DINOv2 inference
  submission_ensemble.py        # weighted DINOv2 + DeepLabV3+ ensemble submission
tools/
  visualize_predictions.py      # overlay predicted masks on images locally
notebooks/
  main.ipynb              # exploration notebook
```

> Model weights (`*.pth`), datasets, and submission CSVs are excluded via `.gitignore` — only source code is tracked.

## Setup

```bash
git clone https://github.com/Saurabh6372/ImageForgery.git
cd ImageForgery
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the Recod.ai dataset into `./recodai-luc-scientific-image-forgery-detection/` (or update the paths in each script's `Config` class), then run a training script from `training/` or an inference script from `inference/`.

## Stack

Python · PyTorch · segmentation_models_pytorch · Hugging Face (DINOv2) · OpenCV · NumPy
