"""Inference module for document layout detection."""

from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import DetrForObjectDetection, DetrImageProcessor

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

# colors for each class when drawing boxes
CLASS_COLORS = [
    "#FF6B6B",  # Caption - red
    "#4ECDC4",  # Footnote - teal
    "#45B7D1",  # Formula - blue
    "#96CEB4",  # List-item - green
    "#FFEAA7",  # Page-footer - yellow
    "#DDA0DD",  # Page-header - plum
    "#98D8C8",  # Picture - mint
    "#F7DC6F",  # Section-header - gold
    "#BB8FCE",  # Table - purple
    "#85C1E9",  # Text - light blue
    "#F0B27A",  # Title - orange
]


class LayoutDetector:
    """Loads a trained DETR model and runs inference on document images."""

    def __init__(self, checkpoint_path, device=None):
        """Load model from checkpoint.

        Args:
            checkpoint_path: path to the lightning checkpoint file
            device: torch device, auto-detected if None
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-50", size={"height": 800, "width": 800}
        )

        # load model from checkpoint
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=len(CLASS_NAMES),
            num_queries=100,
            ignore_mismatched_sizes=True,
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        # remove the 'model.' prefix that lightning adds
        cleaned = {}
        for key, value in state_dict.items():
            clean_key = key.replace("model.", "", 1) if key.startswith("model.") else key
            cleaned[clean_key] = value

        self.model.load_state_dict(cleaned, strict=False)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {checkpoint_path} on {self.device}")

    def predict(self, image, threshold=0.5):
        """Run detection on a single image.

        Args:
            image: PIL Image or path to image file
            threshold: confidence threshold for detections

        Returns:
            list of dicts with keys: box, label, score, class_name
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_size = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_size, threshold=threshold
        )[0]

        detections = []
        for score, label, box in zip(
            results["scores"].cpu(), results["labels"].cpu(), results["boxes"].cpu()
        ):
            label_idx = label.item()
            if label_idx >= len(CLASS_NAMES):
                continue
            detections.append(
                {
                    "box": [round(c, 1) for c in box.tolist()],
                    "label": label_idx,
                    "score": round(score.item(), 3),
                    "class_name": CLASS_NAMES[label_idx],
                }
            )

        # sort by confidence
        detections.sort(key=lambda x: x["score"], reverse=True)
        return detections

    def draw_detections(self, image, detections):
        """Draw bounding boxes on the image.

        Args:
            image: PIL Image
            detections: list of dicts from predict()

        Returns:
            PIL Image with boxes drawn
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()

        for det in detections:
            box = det["box"]
            color = CLASS_COLORS[det["label"]]
            label_text = f"{det['class_name']} {det['score']:.2f}"

            # draw box
            draw.rectangle(box, outline=color, width=2)

            # draw label background
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            draw.rectangle(
                [box[0], box[1] - text_h - 4, box[0] + text_w + 4, box[1]],
                fill=color,
            )
            draw.text((box[0] + 2, box[1] - text_h - 2), label_text, fill="white", font=font)

        return image
