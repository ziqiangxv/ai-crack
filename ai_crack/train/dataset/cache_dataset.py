# -*- coding:utf-8 -*-

''''''

import typing
import torch
from torch.utils.data import Dataset
from functools import lru_cache

from .image_dataset_base import ImageDatasetBase
from .image3d_dataset import Image3DDataset, SlideWindowImage3DDataset, RandomWindowImage3DDataset
from .slice_dataset import (
        SliceDataset,
        Image3DWithGTSliceDataset,
        SlideWindowImage3DWithGTSliceDataset,
        RandomWindowImage3DWithGTSliceDataset
    )


class CachedDataset(Dataset):
    def __init__(self, original_dataset: Dataset, cache_size: int = 100):
        self.original_dataset = original_dataset

        self.__getitem__ = lru_cache(maxsize=cache_size)(self.__getitem__.__wrapped__)

        if hasattr(self.original_dataset, 'collate_fn'):
            self.collate_fn = lambda batch: self.original_dataset.collate_fn(batch)

    @lru_cache(maxsize=100)
    def __getitem__(self, index):
        return self.original_dataset[index]

    def __len__(self):
        return len(self.original_dataset)


class CacheDataReader(Dataset):
    ''''''

    def __init__(self, original_dataset: ImageDatasetBase, cache_size: int = 100):
        ''''''

        self.original_dataset = original_dataset

        if original_dataset.transform is None or not torch.is_grad_enabled():
            self.original_dataset.read_data_with_transform = \
                lru_cache(maxsize = cache_size)(self.original_dataset.read_data_with_transform)

        else:
            self.original_dataset.read_data = lru_cache(maxsize = cache_size)(self.original_dataset.read_data)

        if hasattr(self.original_dataset, 'collate_fn'):
            self.collate_fn = lambda batch: self.original_dataset.collate_fn(batch)

    def __getitem__(self, index):
        return self.original_dataset[index]

    def __len__(self):
        return len(self.original_dataset)


CachedImage3DDataset = lambda cache_size, *args, **kwargs: CachedDataset(Image3DDataset(*args, **kwargs), cache_size)

CachedSlideWindowImage3DDataset = lambda cache_size, *args, **kwargs: \
                                    CachedDataset(SlideWindowImage3DDataset(*args, **kwargs), cache_size)

CachedRandomWindowImage3DDataset = lambda cache_size, *args, **kwargs: \
                                    CacheDataReader(RandomWindowImage3DDataset(*args, **kwargs), cache_size)

CachedSliceDataset = lambda cache_size, *args, **kwargs: CachedDataset(SliceDataset(*args, **kwargs), cache_size)

CachedImage3DWithGTSliceDataset = lambda cache_size, *args, **kwargs: \
                                    CachedDataset(Image3DWithGTSliceDataset(*args, **kwargs), cache_size)

CachedSlideWindowImage3DWithGTSliceDataset = lambda cache_size, *args, **kwargs: \
                                    CachedDataset(SlideWindowImage3DWithGTSliceDataset(*args, **kwargs), cache_size)

CachedRandomWindowImage3DWithGTSliceDataset = lambda cache_size, *args, **kwargs: \
                                    CacheDataReader(RandomWindowImage3DWithGTSliceDataset(*args, **kwargs), cache_size)
