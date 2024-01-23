import cv2
import os
from pathlib import Path
from typing import List, Tuple
from random import Random
import itertools
from collections import deque
from .dataset.coco_dataset import COCODataset

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class COCODatamodule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        # TODO
        pass

    def setup(self, stage=None):
        # Initialize COCODataset for train, val, and test sets
        self.train_dataset = COCODataset(self.data_dir, split="train")
        self.val_dataset = COCODataset(self.data_dir, split="val")
        self.test_dataset = COCODataset(self.data_dir, split="test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
