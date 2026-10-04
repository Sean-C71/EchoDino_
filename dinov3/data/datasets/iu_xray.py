# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import os
import warnings
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from gzip import GzipFile
from io import BytesIO
from mmap import ACCESS_READ, mmap
from typing import Any, Callable, List, Optional, Set, Tuple

import numpy as np

from .extended import ExtendedVisionDataset
from glob import glob
from PIL import Image


class IUXray(ExtendedVisionDataset):

    def __init__(
            self,
            *,
            root: str,
            extra: str,
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            mmap_cache_size: int = 16,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self._entries = glob(f"{root}/*.png")[:100]
        self.transform=transform
        self.target_transform=target_transform

    def get_image_data(self, index: int) -> bytes:
        entry = self._entries[index]
        data = Image.open(entry).convert(mode="RGB")
        return data

    def get_target(self, index: int) -> Any:
        return 0

    def get_targets(self) -> np.ndarray:
        return 0

    def get_class_id(self, index: int) -> str:
        return 0

    def get_class_ids(self) -> np.ndarray:
        return 0

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = self.get_image_data(index)
        target = self.get_target(index)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target=self.target_transform(target)

        return image, target

    def __len__(self) -> int:
        return len(self._entries)
