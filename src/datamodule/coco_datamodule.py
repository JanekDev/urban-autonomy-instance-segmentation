import pathlib
from omegaconf import DictConfig

import torch
import torch.utils.data

from torchvision import datasets, tv_tensors
from torchvision.transforms import v2

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class COCODatamodule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, cfg: DictConfig = None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.config = cfg

    def _prepare_augmentations(self, transforms_list):
        if self.config.transforms.SanitizeBoundingBoxes:
            transforms_list.append(v2.SanitizeBoundingBoxes())
        if self.config.transforms.RandomPhotometricDistort.enabled:
            transforms_list.append(
                v2.RandomPhotometricDistort(
                    p=self.config.transforms.RandomPhotometricDistort.probability
                )
            )
        if self.config.transforms.RandomZoomOut.enabled:
            transforms_list.append(
                v2.RandomZoomOut(
                    fill=self.config.transforms.RandomZoomOut.fill,
                )
            )
        if self.config.transforms.RandomIoUCrop.enabled:
            transforms_list.append(v2.RandomIoUCrop())
        if self.config.transforms.RandomHorizontalFlip.enabled:
            transforms_list.append(
                v2.RandomHorizontalFlip(
                    p=self.config.transforms.RandomHorizontalFlip.probability
                )
            )
        return transforms_list

    def setup(self, stage=None):
        transforms_list = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
        test_transforms = v2.Compose(transforms_list)
        transforms_list = self._prepare_augmentations(transforms_list)
        train_transforms = v2.Compose(transforms_list)
        self.train_dataset = datasets.CocoDetection(
            root=pathlib.Path(self.data_dir) / "train2017",
            annFile=pathlib.Path(self.data_dir)
            / "annotations/instances_train2017.json",
            transforms=train_transforms,
        )
        self.val_dataset = datasets.CocoDetection(
            root=pathlib.Path(self.data_dir) / "val2017",
            annFile=pathlib.Path(self.data_dir) / "annotations/instances_val2017.json",
            transforms=test_transforms,
        )

        self.train_dataset = datasets.wrap_dataset_for_transforms_v2(
            self.train_dataset, target_keys=["boxes", "labels", "masks"]
        )

        self.val_dataset = datasets.wrap_dataset_for_transforms_v2(
            self.val_dataset, target_keys=["boxes", "labels", "masks"]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        return tuple(zip(*batch))
