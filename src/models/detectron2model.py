from typing import Any, Optional
import pytorch_lightning
import torch
from torch import nn
from torch.optim import Optimizer
from detectron2.modeling import build_model
from detectron2 import model_zoo
from detectron2.config import get_cfg


class Detectron2Model(pytorch_lightning.LightningModule):
    def __init__(
        self,
        model: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        optimizer: Optional[Optimizer] = None,
        lr: float = 1e-3,
        pretrained: bool = True,
    ):
        super().__init__()
        self.model = self._get_model(model, pretrained)

    def _get_model(self, name: str, pretrained: bool = True) -> Any:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(name))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(name) if pretrained else ""
        return build_model(cfg)

    def forward(self, inputs, target):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        return self.optimizer


if __name__ == "__main__":
    mynet = Detectron2Model()
    print(mynet)
