"""DETR lightning module for training and validation."""

import pytorch_lightning as pl
import torch
from torchmetrics.detection import MeanAveragePrecision
from transformers import DetrForObjectDetection, DetrImageProcessor


class DETRLightningModule(pl.LightningModule):
    """Wraps DETR model for use with pytorch lightning trainer."""

    def __init__(
        self,
        model_name,
        num_classes,
        learning_rate,
        lr_backbone,
        weight_decay,
        num_queries=100,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

        # load pretrained detr and replace classification head for our classes
        self.model = DetrForObjectDetection.from_pretrained(
            model_name,
            num_labels=num_classes,
            num_queries=num_queries,
            ignore_mismatched_sizes=True,
        )

        self.processor = DetrImageProcessor.from_pretrained(model_name)

        # for computing mAP on validation
        self.val_map = MeanAveragePrecision(iou_type="bbox")

        # keep track of losses for plotting later
        self.training_losses = []
        self.validation_losses = []

    def forward(self, pixel_values, pixel_mask=None, labels=None):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            pixel_values=batch["pixel_values"],
            pixel_mask=batch["pixel_mask"],
            labels=batch["labels"],
        )

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/loss_ce", loss_dict["loss_ce"])
        self.log("train/loss_bbox", loss_dict["loss_bbox"])
        self.log("train/loss_giou", loss_dict["loss_giou"])

        self.training_losses.append(loss.detach().item())
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            pixel_values=batch["pixel_values"],
            pixel_mask=batch["pixel_mask"],
            labels=batch["labels"],
        )

        loss = outputs.loss
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.validation_losses.append(loss.detach().item())

        # get predictions and update mAP
        self._update_map(outputs, batch)

    def _update_map(self, outputs, batch):
        """Convert model outputs to torchmetrics format for mAP calculation."""
        # figure out image sizes from the pixel masks
        target_sizes = []
        for mask in batch["pixel_mask"]:
            h = mask.sum(dim=0).max()
            w = mask.sum(dim=1).max()
            target_sizes.append(torch.tensor([h, w]))
        target_sizes = torch.stack(target_sizes).to(self.device)

        # post process to get actual boxes and scores
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.3
        )

        preds = []
        for result in results:
            preds.append(
                {
                    "boxes": result["boxes"].detach().cpu(),
                    "scores": result["scores"].detach().cpu(),
                    "labels": result["labels"].detach().cpu(),
                }
            )

        # convert ground truth boxes from normalized cxcywh to absolute xyxy
        targets = []
        for label in batch["labels"]:
            boxes_cxcywh = label["boxes"]
            boxes_abs = boxes_cxcywh.clone()
            boxes_abs[:, 0] *= 800
            boxes_abs[:, 1] *= 800
            boxes_abs[:, 2] *= 800
            boxes_abs[:, 3] *= 800

            # cxcywh to xyxy conversion
            boxes_xyxy = torch.zeros_like(boxes_abs)
            boxes_xyxy[:, 0] = boxes_abs[:, 0] - boxes_abs[:, 2] / 2
            boxes_xyxy[:, 1] = boxes_abs[:, 1] - boxes_abs[:, 3] / 2
            boxes_xyxy[:, 2] = boxes_abs[:, 0] + boxes_abs[:, 2] / 2
            boxes_xyxy[:, 3] = boxes_abs[:, 1] + boxes_abs[:, 3] / 2

            targets.append(
                {
                    "boxes": boxes_xyxy.detach().cpu(),
                    "labels": label["class_labels"].detach().cpu(),
                }
            )

        self.val_map.update(preds, targets)

    def on_validation_epoch_end(self):
        metrics = self.val_map.compute()
        self.log("val/mAP", metrics["map"], prog_bar=True)
        self.log("val/mAP_50", metrics["map_50"])
        self.log("val/mAP_75", metrics["map_75"])
        self.log("val/precision", metrics["map"])
        self.log("val/recall", metrics["mar_100"])
        self.val_map.reset()

    def configure_optimizers(self):
        # backbone gets smaller lr since its already pretrained
        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.lr_backbone},
                {"params": other_params, "lr": self.learning_rate},
            ],
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
