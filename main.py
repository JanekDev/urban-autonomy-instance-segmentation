import os

import hydra

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    ModelSummary,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import NeptuneLogger
from torch.distributed.algorithms.ddp_comm_hooks import (
    default_hooks as default,
)

# import model
# import datamodules


@hydra.main(version_base=None, config_path="./configs/")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(seed=cfg.main.seed)

    datamodule = None  # TODO also implement overfit 1 batch

    if cfg.model.restore_from_ckpt is not None:
        print("Restoring entire state from checkpoint...")
        model = None  # TODO
    else:
        print("Creating new model...")
        model = None  # TODO

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

    if not cfg.debug:
        logger = NeptuneLogger(
            api_key=os.environ["WANDB_API_TOKEN"],  # TODO
            project="visionaries2137/urban-autonomy-instance-segmentation",
            log_model_checkpoints=True,
        )
    else:
        logger = None

    callbacks = [
        checkpoint_callback,
        model_summary_callback,
        early_stopping_callback,
        lr_monitor,
    ]

    torch.set_float32_matmul_precision("medium")
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

    if not cfg.test_only:
        trainer.fit(model, datamodule)
        trainer.test(model, datamodule, ckpt_path="best")
    else:
        assert cfg.restore_from_ckpt is not None
        trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
