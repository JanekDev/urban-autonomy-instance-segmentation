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

from src.models.maskrcnn import MaskRCNN

from src.datamodule.coco_datamodule import COCODataModule


def create_model(optimizer, lr, pretrained, restore_from_ckpt):
    # optimizer
    if optimizer == "sgd":
        optimizer = torch.optim.SGD
    elif optimizer == "adam":
        optimizer = torch.optim.Adam
    elif optimizer == "adamw":
        optimizer = torch.optim.AdamW
    else:
        raise ValueError("Invalid optimizer")

    model = MaskRCNN(
        optimizer=optimizer,
        lr=lr,
        pretrained=pretrained,
        restore_from_ckpt=restore_from_ckpt,
    )
    return model


@hydra.main(version_base=None, config_path="./configs/", config_name="default")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(seed=cfg.main.seed)
    torch.set_float32_matmul_precision("medium")

    datamodule = COCODataModule(
        data_dir=cfg.data.data_path,
        batch_size=cfg.data.batch_size,
        data_subset=cfg.data.subset,
        sanitize_bb=cfg.transforms.sanitize_bb,
        rpdist=cfg.transforms.rpdist,
        rzout=cfg.transforms.rzout,
        rioucrop=cfg.transforms.rioucrop,
        rhflip=cfg.transforms.rhflip,
        overfit_batch=cfg.main.overfit_batch,
        urban=cfg.data.urban,
        workers=cfg.data.workers,
    )

    model = create_model(
        optimizer=cfg.model.optimizer,
        lr=cfg.model.lr,
        pretrained=cfg.model.pretrained,
        restore_from_ckpt=cfg.model.restore_from_ckpt,
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
        hparams = OmegaConf.to_container(cfg)
        logger.log_hyperparams(hparams)
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
        max_epochs=cfg.training.epochs,
        sync_batchnorm=True if torch.cuda.is_available() else False,
        log_every_n_steps=10,
    )
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
