import pathlib

import torch
import torch.utils.data

from torchvision import datasets, tv_tensors
from torchvision.transforms import v2

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class COCODatamodule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.RandomPhotometricDistort(p=1),
                v2.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0}),
                v2.RandomIoUCrop(),
                v2.RandomHorizontalFlip(p=1),
                v2.SanitizeBoundingBoxes(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        self.train_dataset = datasets.CocoDetection(
            root=pathlib.Path(self.data_dir) / "train2017",
            annFile=pathlib.Path(self.data_dir)
            / "annotations/instances_train2017.json",
            transforms=transforms,
        )
        self.val_dataset = datasets.CocoDetection(
            root=pathlib.Path(self.data_dir) / "val2017",
            annFile=pathlib.Path(self.data_dir) / "annotations/instances_val2017.json",
            transforms=transforms,
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
