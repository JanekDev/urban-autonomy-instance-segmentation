import pathlib
from omegaconf import DictConfig

import torch
import torch.utils.data

from torchvision import datasets, tv_tensors
from torchvision.transforms import v2

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Tuple, List


class COCODatamodule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        data_subset: float = 1.0,
        sanitize_bb: bool = True,
        rpdist: Tuple[bool, float] = (True, 1),
        rzout: Tuple[bool, List[int]] = (True, (123, 117, 104)),
        rioucrop: Tuple[bool] = (True,),
        rhflip: Tuple[bool, float] = (True, 1),
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data_subset = data_subset
        self.sanitize_bb = sanitize_bb
        self.rpdist = rpdist
        self.rzout = rzout
        self.rioucrop = rioucrop
        self.rhflip = rhflip

    def _prepare_augmentations(self, transforms_list):
        if self.sanitize_bb:
            transforms_list.append(v2.SanitizeBoundingBoxes())
        if self.rpdist[0]:
            transforms_list.append(v2.RandomPhotometricDistort(p=self.rpdist[1]))
        if self.rzout[0]:
            transforms_list.append(
                v2.RandomZoomOut(
                    fill={tv_tensors.Image: self.rzout[1], "others": 0},
                )
            )
        if self.rioucrop[0]:
            transforms_list.append(v2.RandomIoUCrop())
        if self.rhflip[0]:
            transforms_list.append(v2.RandomHorizontalFlip(p=self.rhflip[1]))
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

        if self.data_subset < 1.0:
            self.train_dataset = torch.utils.data.Subset(
                self.train_dataset,
                torch.randperm(len(self.train_dataset))[
                    : int(len(self.train_dataset) * self.data_subset)
                ],
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        return (
            DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
            )
            if self.main.overfit_batch
            else DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
            )
        )

    def collate_fn(self, batch):
        return tuple(zip(*batch))


if __name__ == "__main__":
    datamod = COCODatamodule(data_dir="data/", batch_size=7, data_subset=0.05)
    datamod.setup()
    for batch in datamod.train_dataloader():
        print(batch[1][0])
        break
