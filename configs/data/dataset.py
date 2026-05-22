"""Dataset for DocLayNet."""

import json
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class DocLayNetDataset(Dataset):
    """DocLayNet dataset."""

    def __init__(self, data_dir, split="train", image_size=640):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size

        # Load annotations
        ann_file = self.data_dir / f"{split}.json"
        with open(ann_file) as f:
            self.coco = json.load(f)

        self.images = self.coco["images"]
        self.annotations = self._group_by_image()

    def _group_by_image(self):
        """Group annotations by image id."""
        grouped = {}
        for ann in self.coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in grouped:
                grouped[img_id] = []
            grouped[img_id].append(ann)
        return grouped

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info["id"]

        # Load image
        img_path = self.data_dir / "images" / img_info["file_name"]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))

        # Normalize and convert to tensor
        image = image / 255.0
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        # Get bboxes and labels
        anns = self.annotations.get(img_id, [])
        bboxes = [ann["bbox"] for ann in anns]
        labels = [ann["category_id"] for ann in anns]

        return {
            "image": image,
            "bboxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch):
    """Collate function for dataloader."""
    images = torch.stack([item["image"] for item in batch])
    bboxes = [item["bboxes"] for item in batch]
    labels = [item["labels"] for item in batch]
    return {"images": images, "bboxes": bboxes, "labels": labels}
