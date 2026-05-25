"""Dataset module for DocLayNet."""

import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import DetrImageProcessor

CLASS_NAMES = [
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
]


class DocLayNetDataset(Dataset):
    """DocLayNet dataset that loads COCO annotations for DETR."""

    def __init__(self, annotation_file, image_dir, processor, subset_ratio=1.0):
        self.image_dir = Path(image_dir)
        self.processor = processor

        with open(annotation_file) as file:
            coco_data = json.load(file)

        self.images = coco_data["images"]

        if subset_ratio < 1.0:
            sample_size = max(1, int(len(self.images) * subset_ratio))
            self.images = self.images[:sample_size]

        # build a dict so we can quickly find annotations for each image
        self.annotations = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        # category ids in coco start from 1 but we need 0-indexed
        self.cat_mapping = {}
        for cat in coco_data["categories"]:
            if cat["name"] in CLASS_NAMES:
                self.cat_mapping[cat["id"]] = CLASS_NAMES.index(cat["name"])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        image_id = img_info["id"]

        img_path = self.image_dir / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")

        raw_anns = self.annotations.get(image_id, [])

        # prepare annotations in the format DETR processor expects
        formatted_anns = []
        for ann in raw_anns:
            cat_id = ann["category_id"]
            if cat_id not in self.cat_mapping:
                continue
            bbox = ann["bbox"]
            area = bbox[2] * bbox[3]
            if area <= 0:
                continue
            formatted_anns.append(
                {
                    "bbox": bbox,
                    "category_id": self.cat_mapping[cat_id],
                    "area": area,
                    "iscrowd": 0,
                }
            )

        target = {"image_id": image_id, "annotations": formatted_anns}

        encoding = self.processor(images=image, annotations=target, return_tensors="pt")

        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"][0]

        return {"pixel_values": pixel_values, "labels": labels}


def collate_fn(batch):
    """Pads images to same size so we can stack them into a batch."""
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]

    max_h = max(pv.shape[1] for pv in pixel_values)
    max_w = max(pv.shape[2] for pv in pixel_values)

    padded_images = []
    pixel_masks = []
    for pv in pixel_values:
        height, width = pv.shape[1], pv.shape[2]
        padded = torch.zeros(3, max_h, max_w, dtype=pv.dtype)
        padded[:, :height, :width] = pv
        padded_images.append(padded)

        # mask shows which pixels are real (1) and which are padding (0)
        mask = torch.zeros(max_h, max_w, dtype=torch.long)
        mask[:height, :width] = 1
        pixel_masks.append(mask)

    return {
        "pixel_values": torch.stack(padded_images),
        "pixel_mask": torch.stack(pixel_masks),
        "labels": labels,
    }


class DocLayNetDataModule(pl.LightningDataModule):
    """Handles train/val dataloaders for lightning trainer."""

    def __init__(self, data_dir, model_name, batch_size, num_workers, subset_ratio=1.0):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.subset_ratio = subset_ratio

        self.processor = DetrImageProcessor.from_pretrained(
            model_name, size={"height": 800, "width": 800}
        )

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = DocLayNetDataset(
                annotation_file=self.data_dir / "COCO" / "train.json",
                image_dir=self.data_dir / "PNG",
                processor=self.processor,
                subset_ratio=self.subset_ratio,
            )
            self.val_dataset = DocLayNetDataset(
                annotation_file=self.data_dir / "COCO" / "val.json",
                image_dir=self.data_dir / "PNG",
                processor=self.processor,
                subset_ratio=self.subset_ratio,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
