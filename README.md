# Document Layout Detection with DETR

Detecting layout elements in document images using DETR fine-tuned on DocLayNet.

## Problem Statement

In this project I detect layout elements in document images. This means finding where tables, titles, figures, text blocks are on a page. This is useful in OCR systems - you first need to know where elements are before reading text. For example if you find a table region you can use special table extraction.

OCR tools like Tesseract and PaddleOCR need to know where things are on the page before reading text. My model can be used for this step.

## Input and Output

Input is a document image in PNG or JPG format, resized to 800x800 pixels. Output is bounding box coordinates, class name and confidence score for each detected element.

### Classes (11 total)

Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, Title

## Results

Trained on 10% subset (~6900 images), 29 epochs on Google Colab T4:

| Metric           | Before Training | After Training    |
| ---------------- | --------------- | ----------------- |
| mAP@0.5          | 0.155           | **0.456**         |
| Recall (mAR@100) | 0.173           | **0.405**         |
| F1-score         | 0.164           | **0.429**         |
| Val Loss         | 1.820           | **0.930**         |
| Inference time   | -               | **~4500ms** (CPU) |

The main metric mAP@0.5 reached 0.456 using only 10% of the dataset. The original DocLayNet paper reports 0.57 mAP with Deformable DETR trained on the full dataset. GPU inference is estimated ~100ms on T4.

For baseline comparison, the DocLayNet paper reports 0.363 mAP with Faster R-CNN on the full dataset. LayoutParser with Detectron2 pretrained on PubLayNet was considered as baseline but direct comparison was not meaningful due to subset training and dataset differences.

## Dataset

DocLayNet from IBM Research, published at KDD 2022. The dataset has about 80 thousand document pages with different types like financial reports, scientific papers, legal documents and manuals. Annotations are in COCO format. I use 10% subset because the full dataset is too large for free Colab.

I save train/val split IDs as JSON files for reproducibility instead of using random seed which gives different results on different CPUs.

Paper: https://arxiv.org/abs/2206.01062

## Model

I use DETR (facebook/detr-resnet-50) pretrained on COCO and fine-tune on DocLayNet with PyTorch Lightning. Training uses AdamW optimizer with learning rate 0.0001 for model and 0.00001 for backbone, batch size 4, 30 epochs with early stopping.

## Setup

```bash
git clone https://github.com/alperendolunayy/document-layout-detection.git
cd document-layout-detection
pip install poetry
poetry install
```

## Download Data and Model

```bash
# download model checkpoint via DVC (opens browser for Google Drive authentication)
poetry run dvc pull

# download dataset
poetry run python -m docuvision.commands download
```

On first run, DVC will open your browser for Google Drive OAuth. Click "Advanced" and allow access to download the checkpoint. The first prediction will also download DETR model architecture and image processor from HuggingFace (~170MB), the fine-tuned weights are loaded from the DVC checkpoint on top.

## Train

```bash
poetry run python -m docuvision.commands train --subset_ratio=0.1 --max_epochs=30
```

Training uses PyTorch Lightning, Hydra configs, and logs to MLflow. Checkpoints are saved automatically and training can resume from last checkpoint.

## Predict

```bash
poetry run python -m docuvision.commands predict --image_path path/to/image.png
```

Saves annotated image with colored bounding boxes and a JSON file with detections.

## Gradio Demo

```bash
poetry run python -m docuvision.commands demo
```

Opens a web interface at `http://127.0.0.1:8080` where you can drag and drop document images and adjust the confidence threshold.

## FastAPI

```bash
poetry run python -m docuvision.commands serve
```

- `GET /health` - check if service is running
- `POST /detect` - upload image and get JSON response
- API docs at `http://127.0.0.1:8000/docs`

## Docker

```bash
# make sure checkpoint is downloaded first (dvc pull)
docker build -t docuvision .
docker run -p 8080:8080 docuvision
```

## Project Structure

```
docuvision/
├── commands.py            # CLI commands (train, predict, demo, serve)
├── inference.py           # Model loading and prediction
├── demo.py                # Gradio web interface
├── api.py                 # FastAPI REST endpoints
├── data/
│   └── dataset.py         # Dataset loader with split saving
└── training/
    ├── lightning_module.py # DETR wrapper for Lightning
    └── train.py           # Training with MLflow logging
configs/
├── config.yaml
├── data/doclaynet.yaml
├── model/detr.yaml
├── serving/default.yaml
└── training/default.yaml
```

## Tools

- **Model**: DETR (facebook/detr-resnet-50)
- **Training**: PyTorch Lightning
- **Config**: Hydra
- **Logging**: MLflow
- **Data**: DVC (Google Drive)
- **API**: FastAPI
- **Demo**: Gradio
- **Code Quality**: Ruff + pre-commit
- **Dependencies**: Poetry
- **Container**: Docker
