# Document Layout Detection

This project detects document layout elements using YOLOv8.

## About

I am using DocLayNet dataset to train a YOLOv8 model for detecting different parts of documents like tables, figures, text blocks etc.

### Classes

The model detects these 11 classes:
- Caption
- Footnote
- Formula
- List-item
- Page-footer
- Page-header
- Picture
- Section-header
- Table
- Text
- Title

## Dataset

DocLayNet from IBM Research. It has around 80K document images.

Paper: https://arxiv.org/abs/2206.01062

## Setup

```bash
# create virtual environment
python -m venv .venv
source .venv/bin/activate  # linux/mac
# .venv\Scripts\activate  # windows

# install dependencies
pip install -e .
```

## Train

```bash
python -m docuvision.commands train
```

## Predict

```bash
python -m docuvision.commands predict --image_path path/to/image.png
```
