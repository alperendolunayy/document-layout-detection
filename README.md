# Document Layout Detection with DETR

Detecting layout elements in document images using DETR fine-tuned on DocLayNet.

## About

I detect things like tables, figures, text blocks, headers in document pages. This is useful for OCR pipelines - you need to know where elements are before reading text. I use DETR (facebook/detr-resnet-50) and fine-tune it on DocLayNet dataset.

### Classes (11 total)

Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, Title

### Training Results

Trained on 10% subset (~6900 images), 29 epochs on Google Colab T4:

- mAP: 0.295
- Val Loss: 0.93
- Train Loss: 0.56

Paper baseline (Deformable DETR, full dataset): 0.57 mAP

## Setup

```bash
git clone https://github.com/alperendolunayy/document-layout-detection.git
cd document-layout-detection
pip install poetry
poetry install
```

## Train

```bash
poetry run python -m docuvision.commands train --subset_ratio=0.1 --max_epochs=30
```

You need to download DocLayNet dataset first and put it in `data/doclaynet/`.

Training uses PyTorch Lightning, Hydra configs, and logs to MLflow. Checkpoints are saved automatically and training can resume from last checkpoint.

## Predict

```bash
poetry run python -m docuvision.commands predict --image_path path/to/image.png --checkpoint checkpoints/last.ckpt
```

Saves annotated image with colored bounding boxes and a JSON file with detections.

## Dataset

DocLayNet from IBM Research. ~80K document pages in COCO format.

I save train/val split IDs as JSON files for reproducibility (instead of using random seed which gives different results on different CPUs).

Paper: https://arxiv.org/abs/2206.01062

## Project Structure

```
docuvision/
├── commands.py            # CLI commands (train, predict)
├── inference.py           # Model loading and prediction
├── data/
│   └── dataset.py         # Dataset loader with split saving
└── training/
    ├── lightning_module.py # DETR wrapper for Lightning
    └── train.py           # Training with MLflow logging
configs/
├── config.yaml
├── data/doclaynet.yaml
├── model/detr.yaml
└── training/default.yaml
```
