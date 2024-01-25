from typing import Any, Optional
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights
from ..metrics.metrics import AveragePrecision


class MaskRCNN(pl.LightningModule):
    def __init__(
        self,
        optimizer: Optional[Optimizer] = None,
        lr: float = 1e-3,
        pretrained: bool = True,
        restore_from_ckpt: Optional[str] = None,
    ):
        super().__init__()
        self.model = self._get_model(
            "maskrcnn_resnet50_fpn", pretrained, weights=MaskRCNN_ResNet50_FPN_Weights
        )
        self.optimizer = optimizer or Adam(self.model.parameters(), lr=self.hparams.lr)
        self.lr = lr
        self.pretrained = pretrained
        self.restore_from_ckpt = restore_from_ckpt

    def _get_model(
        self, name: str, pretrained: bool = True, weights=MaskRCNN_ResNet50_FPN_Weights
    ) -> Any:
        model = (
            maskrcnn_resnet50_fpn()
            if not pretrained
            else maskrcnn_resnet50_fpn(weights=weights)
        )
        return model

    def setup(self, stage: Optional[str] = None):
        self.model.to(self.device)
        self.metric1 = AveragePrecision(num_classes=80, iou_threshold=0.5)
        self.metric2 = AveragePrecision(num_classes=80, calculate_full_ap=True)

    def forward(self, inputs, target=None):
        self.model.eval()
        outputs = self.model(inputs, targets=target)
        return outputs

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        self.log("train_compound_loss", total_loss)
        self.log("train_loss_classifier", loss_dict["loss_classifier"])
        self.log("train_loss_box_reg", loss_dict["loss_box_reg"])
        self.log("train_loss_mask", loss_dict["loss_mask"])
        self.log("train_loss_objectness", loss_dict["loss_objectness"])
        self.log("train_loss_rpn_box_reg", loss_dict["loss_rpn_box_reg"])
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        pred_dict = self.model(images, targets)  # for metrics
        self.metric(pred_dict["masks"], targets["masks"])
        self.metric2(pred_dict["masks"], targets["masks"])
        self.model.train()  # HACK it is safe to do this here
        loss_dict = self.model(images, targets)  # for model monitoring
        total_loss = sum(loss for loss in loss_dict.values())
        self.model.eval()
        self.log("val_compound_loss", total_loss)
        self.log("val_loss_classifier", loss_dict["loss_classifier"])
        self.log("val_loss_box_reg", loss_dict["loss_box_reg"])
        self.log("val_loss_mask", loss_dict["loss_mask"])
        self.log("val_loss_objectness", loss_dict["loss_objectness"])
        self.log("val_loss_rpn_box_reg", loss_dict["loss_rpn_box_reg"])
        self.log("val_AP50", self.metric.ap_scores.mean())
        self.log("val_AP_full", self.metric2.ap_scores.mean())
        return total_loss

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)
