import os

import hydra

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torchvision import models
from omegaconf import OmegaConf

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    ModelSummary,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger
from torch.distributed.algorithms.ddp_comm_hooks import (
    default_hooks as default,
)

from src.datamodule.coco_datamodule import COCODatamodule


@hydra.main(version_base=None, config_path="./configs/", config_name="default")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(seed=cfg.main.seed)

    datamodule = COCODatamodule(
        data_dir=cfg.data.data_path,
        batch_size=cfg.data.batch_size,
        cfg=cfg,
    )

    if cfg.model.restore_from_ckpt is not None:
        print("Restoring entire state from checkpoint...")
        model = None  # TODO
    else:
        print("Creating new model...")
        model = models.get_model(
            "maskrcnn_resnet50_fpn_v2", weights=None, weights_backbone=None
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"],
        filename="epoch_{epoch}-f1_{val_f1:.2f}",
        monitor=cfg.tracking.monitor,
        auto_insert_metric_name=False,
        verbose=True,
        mode=cfg.tracking.monitor_mode,
    )

    model_summary_callback = ModelSummary(max_depth=1)

    early_stopping_callback = EarlyStopping(
        monitor=cfg.tracking.monitor,
        mode=cfg.tracking.monitor_mode,
        patience=cfg.tracking.es_patience,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    if cfg.main.logging:
        logger = WandbLogger(
            project="urban-autonomy-instance-segmentation", log_model=True
        )
    else:
        logger = None

    callbacks = [
        checkpoint_callback,
        model_summary_callback,
        early_stopping_callback,
        lr_monitor,
    ]
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        devices="auto" if cfg.training.devices <= 0 else cfg.training.devices,
        accelerator="gpu"
        if torch.cuda.is_available() and cfg.training.devices > 0
        else "cpu",
        precision=cfg.training.precision,
        max_epochs=cfg.training.epochs,
        benchmark=True if cfg.training.devices > 0 else False,
        sync_batchnorm=True if torch.cuda.is_available() else False,
    )
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
