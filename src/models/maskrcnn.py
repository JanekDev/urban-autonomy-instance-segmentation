from typing import Any, Optional
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights
from src.datamodule.coco_datamodule import COCODataModule


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

    def forward(self, inputs, target=None):
        self.model.eval()
        outputs = self.model(inputs, targets=target)
        return outputs

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        self.log("train_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        self.model.training = True  # HACK
        loss_dict = self.model(images, targets)
        self.model.training = True  # HACK
        total_loss = sum(loss for loss in loss_dict.values())
        self.log("val_loss", total_loss)
        return total_loss

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)
