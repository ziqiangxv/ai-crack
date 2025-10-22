# -*- coding:utf-8 -*-

''''''

import typing
import torch
from torch.utils.data import Dataset

from .dataset_config import DatasetConfig
from ...tensor_utils import normalize_tensor
from .image_dataset_base import ImageDatasetBase
from ...factory import REGISTER


class Image3DDataset(ImageDatasetBase):
    ''''''

    def __init__(
        self,
        config_path: str,
        reader: typing.Callable[[str], torch.Tensor],
        transform: typing.Optional[typing.Callable[[torch.Tensor], torch.Tensor]] = None,
        normalize: bool = True,
        device: str = 'cuda:0'):
        ''''''

        super().__init__(reader, transform, normalize, device)

        self.config = DatasetConfig(config_path)

    def __len__(self):
        return len(self.config.images)

    def __getitem__(self, idx):
        image_path = self.config.images[idx]
        gt_path = self.config.gts[idx]

        image = self.get_data(image_path, 'image')
        gt = self.get_data(gt_path, 'gt')

        return image, gt


class SlideWindowImage3DDataset(ImageDatasetBase):
    ''''''

    def __init__(
        self,
        config_path: str,
        window_size: typing.Tuple[int, int, int],
        window_overlap: typing.Tuple[int, int, int],
        reader: typing.Callable[[str], torch.Tensor],
        transform: typing.Optional[typing.Callable[[torch.Tensor], torch.Tensor]] = None,
        normalize: bool = True,
        device: str = 'cuda:0'):
        ''''''

        super().__init__(reader, transform, normalize, device)

        self.config = DatasetConfig(config_path)

        stride = (
                window_size[0] - window_overlap[0],
                window_size[1] - window_overlap[1],
                window_size[2] - window_overlap[2],
            )

        gt = self.reader(self.config.gts[0])
        image_size = gt.shape[1:]
        del gt

        self.image_num = len(self.config.gts)
        sample_vois = []
        i_, j_, k_ = 0, 0, 0

        for i in range(0, image_size[0] - window_size[0], stride[0]):
            for j in range(0, image_size[1] - window_size[1], stride[1]):
                for k in range(0, image_size[2] - window_size[2], stride[2]):
                    sample_vois.append([(i, j, k), (i + window_size[0], j + window_size[1], k + window_size[2])])
                    i_, j_, k_ = i, j, k

        if (image_size[0] - window_size[0] - i_) / image_size[0] > 0.05 or \
            (image_size[1] - window_size[1] - j_) / image_size[1] > 0.05 or \
            (image_size[2] - window_size[2] - k_) / image_size[2] > 0.05:
            for i in range(image_size[0] - window_size[0], image_size[0], window_size[0]):
                for j in range(image_size[1] - window_size[1], image_size[1], window_size[1]):
                    for k in range(image_size[2] - window_size[2], image_size[2], window_size[2]):
                        sample_vois.append([(i, j, k), (i + window_size[0], j + window_size[1], k + window_size[2])])

        self.sample_vois = sample_vois

    def __len__(self):
        return len(self.sample_vois) * self.image_num

    def __getitem__(self, idx):
        image_idx = idx // len(self.sample_vois)
        sample_idx = idx % len(self.sample_vois)

        image_path = self.config.images[image_idx]
        gt_path = self.config.gts[image_idx]

        image = self.get_data(image_path, 'image')
        gt = self.get_data(gt_path, 'gt')

        start, end = self.sample_vois[sample_idx]
        image = image[..., start[0] : end[0], start[1] : end[1], start[2] : end[2]]
        gt = gt[..., start[0] : end[0], start[1] : end[1], start[2] : end[2]]

        return image, gt


class RandomWindowImage3DDataset(ImageDatasetBase):
    ''''''

    def __init__(
        self,
        config_path: str,
        window_size: typing.Tuple[int, int, int],
        num_samples_per_image: int,
        reader: typing.Callable[[str], torch.Tensor],
        transform: typing.Optional[typing.Callable[[torch.Tensor], torch.Tensor]] = None,
        normalize: bool = True,
        device: str = 'cuda:0'):
        ''''''

        super().__init__(reader, transform, normalize, device)

        self.config = DatasetConfig(config_path)

        self.num_samples_per_image = num_samples_per_image
        self.window_size = window_size

        gt = self.reader(self.config.gts[0])
        self.image_size = gt.shape[1:]
        del gt
        self.image_num = len(self.config.gt)

    def __len__(self):
        return self.image_num * self.num_samples_per_image

    def __getitem__(self, idx):
        image_idx = idx // self.num_samples_per_image

        image_path = self.config.images[image_idx]
        gt_path = self.config.gts[image_idx]

        image = self.get_data(image_path, 'image')
        gt = self.get_data(gt_path, 'gt')

        z_size, y_size, x_size = self.image.shape
        z_max = z_size - self.window_size[0]
        y_max = y_size - self.window_size[1]
        x_max = x_size - self.window_size[2]

        # 随机选择窗口的起始位置
        z_start = torch.randint(0, z_max)
        y_start = torch.randint(0, y_max)
        x_start = torch.randint(0, x_max)

        window = [
            (z_start, y_start, x_start),
            (z_start + self.window_size[0], y_start + self.window_size[1], x_start + self.window_size[2])
        ]

        # 提取随机窗口
        window_image = image[...,
                            window[0][0] : window[1][0],
                            window[0][1] : window[1][1],
                            window[0][2] : window[1][2]]
        window_gt    = gt[...,
                            window[0][0] : window[1][0],
                            window[0][1] : window[1][1],
                            window[0][2] : window[1][2]]

        self._random_window = window

        return window_image, window_gt


REGISTER('dataset', 'image3d_dataset', Image3DDataset)

REGISTER('dataset', 'slide_window_image3d_dataset', SlideWindowImage3DDataset)

REGISTER('dataset', 'random_window_image3d_dataset', RandomWindowImage3DDataset)
