# Document Layout Detection with DETR

Detecting layout elements in document images using DETR fine-tuned on DocLayNet.

## About

I detect things like tables, figures, text blocks, headers in document pages. This is useful for OCR pipelines - you need to know where elements are before reading text. I use DETR (facebook/detr-resnet-50) and fine-tune it on DocLayNet dataset.

### Classes (11 total)

Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, Title

### Results

Trained on 10% subset (~6900 images), 29 epochs on Google Colab T4:

| Metric           | Before Training | After Training    |
| ---------------- | --------------- | ----------------- |
| mAP@0.5          | 0.155           | **0.456**         |
| Recall (mAR@100) | 0.173           | **0.405**         |
| F1-score         | 0.164           | **0.429**         |
| Val Loss         | 1.820           | **0.930**         |
| Inference time   | -               | **~4500ms** (CPU) |

The main metric mAP@0.5 reached 0.456 using only 10% of the dataset. The original DocLayNet paper reports 0.57 mAP with Deformable DETR trained on the full dataset. GPU inference is estimated ~100ms on T4.

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
- `POST /detect` - upload image and get JSON response with detections
- API docs at `http://127.0.0.1:8000/docs`

## Docker

```bash
docker build -t docuvision .
docker run -p 8080:8080 docuvision
```

## Dataset

DocLayNet from IBM Research. ~80K document pages in COCO format.

I save train/val split IDs as JSON files for reproducibility (instead of using random seed which gives different results on different CPUs).

Paper: https://arxiv.org/abs/2206.01062

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
└── training/default.yaml
```
